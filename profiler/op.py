from typing import Literal
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.fusions.fused_layer_norm import FusedLayerNorm
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.module import MegatronModule
from enum import Enum
from megatron.core import mpu, tensor_parallel
import torch 
from torch import Tensor

from megatron.core.utils import make_viewless_tensor 
from megatron.core.transformer.custom_layers.transformer_engine import (
    TEDotProductAttention,
    TELayerNormColumnParallelLinear,
    TENorm,
    TERowParallelLinear,
)

class OpType(Enum):
    EMBEDDING = 1
    LAYER_NORM_SELF_ATTENTION_DROPOUT = 2
    LAYER_NORM_MLP_DROPOUT = 3
    LAYER_NORM_POST_PROCESS = 4
    LanguageModelEmbedding = 5
    LocalLayerNormSelfAttentionDropout = 6
    LocalLayerNormMlpDropout = 7
    PostProcess = 8
    LocalLayerNorm = 9
    LocalLayerNormPostProcess = 10
    TELayerNormSelfAttentionDropout = 11
    TELayerNormMlpDropout = 12
    TELayerNorm = 13
    TELayerNormPostProcess = 14


class OpModule(MegatronModule):
    def __init__(self, 
               op_type: OpType,
        op_name: str,
        config: TransformerConfig,
               ):
        super().__init__(config)
        self.config = config 
        self.op_type = op_type
        self.op_name = op_name
        self.tp_size = mpu.get_tensor_model_parallel_world_size()
        self.dp_size = mpu.get_data_parallel_world_size()
        self.cp_size = mpu.get_context_parallel_world_size()
        
        # shapes
        self.seq_length = config.seq_length
        self.micro_batch_size = config.micro_batch_size
        self.hidden_size = config.hidden_size
        # [s, b, h]
        self.hidden_state_size = [
            config.seq_length,
            config.micro_batch_size,
            config.hidden_size,
        ]
        

class OpLanguageModelEmbedding(OpModule):
    def __init__(self,
                 op_type: OpType,
                 op_name: str,
                 config: TransformerConfig,
               ) -> None:
        super().__init__(op_type,  op_name, config)
        self.max_sequence_length = config.max_sequence_length
        self.padded_vocab_size = config.padded_vocab_size
        
        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=self.padded_vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type='learned_absolute',
        )
        # self.weight_size = (
        #     config.padded_vocab_size * config.hidden_size
        # ) // self.tp_size
       
        self.weight_size = 0
        
        
        for param in self.embedding.parameters():
            self.weight_size += param.numel()

        self.input_tensors_info = {
            "input_ids": {
                "shape": [self.micro_batch_size, self.seq_length],
                "tp_split_dim": -1,
                "dp_split_dim": -1,
                "cp_split_dim": -1,
            },
            'position_ids': {'shape': [self.micro_batch_size, self.seq_length,], 'tp_split_dim': -1, 'dp_split_dim': -1}
        }
        self.output_tensors_info= {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
                "cp_split_dim": 0
            }
        }
        self.input_extra_tensors_info = {}
        self.output_extra_tensors_info = {}
    def forward(self, 
                input_tensors: dict[str, Tensor],
                input_extra_tensors: dict[str, Tensor],
                output_extra_tensors:  dict[str, Tensor],
                ):
        output_tensors = {}
        input_ids: Tensor = input_tensors["input_ids"]
        position_ids: Tensor = input_tensors["position_ids"]
        decoder_input =  self.embedding(input_ids=input_ids, position_ids=position_ids)
        output_tensors["hidden_states"] = decoder_input
        return output_tensors


    


class OpLocalLayerNormSelfAttentionDropout(OpModule):
    """
    input_layernorm + self_attention + self_attn_bda.

    """

    def __init__(
        self,
        op_type: OpType,
        op_name: str,
        config: TransformerConfig,
    ):
        """
        Args:
            layer_number: The global number of transformer layer, start with 1.
        """
        super().__init__(op_type, op_name, config)
        self.layer_number = 1
        self.hidden_dropout = config.hidden_dropout

        self.input_layernorm = FusedLayerNorm(self.config, self.config.hidden_size, self.config.layernorm_epsilon)
        self.self_attention = SelfAttention(self.config, SelfAttentionSubmodules(
                    linear_qkv=ColumnParallelLinear,
                    core_attention=DotProductAttention,
                    linear_proj=RowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ), 1, AttnMaskType.causal)

        self.self_attn_bda = get_bias_dropout_add

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

        qkv_projection_size = config.kv_channels * config.num_attention_heads
        qkv_weight = config.hidden_size * qkv_projection_size * 3 / self.tp_size
        dense_weight = (
            (config.kv_channels * config.num_attention_heads) * config.hidden_size
        ) / self.tp_size
        self.weight_size = qkv_weight + dense_weight
        self.input_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.output_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.input_extra_tensors_info = {
            "attention_mask": {
                "shape": [
                    config.micro_batch_size // self.dp_size,
                    1,
                    config.seq_length,
                    config.seq_length,
                ],
                "tp_split_dim": -1,
                "dp_split_dim": -1,
                "recv_from": 0,
            }
        }
        self.output_extra_tensors_info = {}
    def forward(
        self,
        input_tensors: dict[str, Tensor],
        input_extra_tensors: dict[str, Tensor],
        output_extra_tensors: dict[str, Tensor],
    ):
        output_tensors = {}
        if type(input_tensors) is list:
            input_tensors = input_tensors[0]
        hidden_states: Tensor = input_tensors["hidden_states"]

        rotary_pos_emb = input_tensors.get("rotary_pos_emb", None)
        inference_params = input_tensors.get("inference_params", None)
        packed_seq_params = input_tensors.get("packed_seq_params", None)

        attention_mask: Tensor = input_extra_tensors["attention_mask"]
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        output_tensors["hidden_states"] = hidden_states
        return output_tensors




class OpLocalLayerNormMlpDropout(OpModule):
    def __init__(
        self,
        op_type: OpType,
        op_name: str,
        config: TransformerConfig,
    ):
        super().__init__(op_type, op_name, config)

        self.layer_nummber = 1 
        self.hidden_dropout = config.hidden_dropout 
        

        self.pre_mlp_layernorm = FusedLayerNorm(self.config, self.config.hidden_size, self.config.layernorm_epsilon)
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = MLP(config, MLPSubmodules(
                linear_fc1=ColumnParallelLinear,
                linear_fc2=RowParallelLinear,
            ))
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(self.layer_number)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = get_bias_dropout_add

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

        gemm_1_weight = config.hidden_size * config.ffn_hidden_size / self.tp_size
        gemm_2_weight = config.ffn_hidden_size * config.hidden_size / self.tp_size
        self.weight_size = gemm_1_weight + gemm_2_weight

        self.input_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.output_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.input_extra_tensors_info = {}
        self.output_extra_tensors_info = {}

    def forward(
        self,
        input_tensors: dict[str, Tensor],
        input_extra_tensors: dict[str, Tensor],
        output_extra_tensors: dict[str, Tensor],
    ):
        output_tensors = {}
        if type(input_tensors) is list:
            input_tensors = input_tensors[0]
        hidden_states: Tensor = input_tensors["hidden_states"]
        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(
                self.training, self.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, self.hidden_dropout)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )
        output_tensors["hidden_states"] = output
        return output_tensors


class OpLocalLayerNormPostProcess(OpModule):
    def __init__(
        self,
        op_type: OpType,
        op_name: str,
        config: TransformerConfig, 
        parallel_output: bool = True,
        num_tokentypes: int = 0,
        fp16_lm_cross_entropy: bool = False
    ):
        super().__init__(op_type, op_name, config)

        self.hidden_dropout = self.config.hidden_dropout

        self.final_layernorm = FusedLayerNorm(self.config, config.hidden_size, config.layernorm_epsilon)
        if config.defer_embedding_wgrad_compute:
            self.embedding_activation_buffer = []
            self.grad_output_buffer = []
        else:
            self.embedding_activation_buffer = None
            self.grad_output_buffer = None

        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy

        self.output_layer = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.padded_vocab_size,
            config=config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not self.parallel_output,
            skip_weight_param_allocation=True,
            embedding_activation_buffer=self.embedding_activation_buffer,
            grad_output_buffer=self.grad_output_buffer,
        )

        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_position_embeddings,
            position_embedding_type='learned_absolute',
            num_tokentypes=num_tokentypes,
        )


        self.weight_size = config.padded_vocab_size * config.hidden_size / self.tp_size

        self.input_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.output_tensors_info = {
            "output_tensor": {"shape": [1], "tp_split_dim": -1, "dp_split_dim": -1}
        }
        self.input_extra_tensors_info = {
            "labels": {
                "shape": [
                    config.micro_batch_size // self.dp_size,
                    config.seq_length
                ],
                "tp_split_dim": -1,
                "dp_split_dim": 0,
                "recv_from": 0,
            }
        }
        self.shared_weights_info = {
            "word_embeddings": {
                "root": False,
                "sharing_with_ops": [0],
                "shape": [config.padded_vocab_size, config.hidden_size],
                "tp_split_dim": 0,
                "dp_split_dim": -1,
            }
        }
        self.output_extra_tensors_info = {}

    def forward(
        self,
        input_tensors: dict[str, Tensor],
        input_extra_tensors: dict[str, Tensor],
        output_extra_tensors: dict[str, Tensor],
    ):
        output_tensors = {}

        if type(input_tensors) is list:
            input_tensors = input_tensors[0]
        hidden_states: Tensor = input_tensors["hidden_states"]

        # Optional Layer norm post the cross-attention.
        final_layernorm_output = self.final_layernorm(hidden_states)

        # always post process
        weights = self.embedding.word_embeddings.weight

        output, _ = self.output_layer(final_layernorm_output, weights)

        labels = input_extra_tensors["labels"]

        if labels is None:
            output_tensors["output_tensor"] = output.transpose(0, 1).contiguous()
            return output_tensors
        else:
            labels = labels.transpose(0, 1).contiguous()
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
            else:
                loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
            loss = loss.transpose(0, 1).contiguous()

        output_tensors["output"] = loss

        return output_tensors
    


class OpPostProcess(OpModule):

    def __init__(
        self,
        op_type: OpType,
        op_name: str,
        config: TransformerConfig, 
        parallel_output: bool = True,
        num_tokentypes: int = 0,
        fp16_lm_cross_entropy: bool = False
    ):
        super().__init__(op_type, op_name, config)

        self.hidden_dropout = self.config.hidden_dropout

        # self.final_layernorm = build_module(
        #     submodules.final_layernorm,
        #     config=self.config,
        #     hidden_size=self.config.hidden_size,
        #     eps=self.config.layernorm_epsilon,
        # )
        if config.defer_embedding_wgrad_compute:
            self.embedding_activation_buffer = []
            self.grad_output_buffer = []
        else:
            self.embedding_activation_buffer = None
            self.grad_output_buffer = None

        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy

        self.output_layer = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.padded_vocab_size,
            config=config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not self.parallel_output,
            skip_weight_param_allocation=True,
            embedding_activation_buffer=self.embedding_activation_buffer,
            grad_output_buffer=self.grad_output_buffer,
        )

        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_position_embeddings,
            position_embedding_type='learned_absolute',
            num_tokentypes=num_tokentypes,
        )


        self.weight_size = config.padded_vocab_size * config.hidden_size / self.tp_size

        self.input_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.output_tensors_info = {
            "output_tensor": {"shape": [1], "tp_split_dim": -1, "dp_split_dim": -1}
        }
        self.input_extra_tensors_info = {
            "labels": {
                "shape": [
                    config.micro_batch_size // self.dp_size,
                    config.seq_length
                ],
                "tp_split_dim": -1,
                "dp_split_dim": 0,
                "recv_from": 0,
            }
        }
        self.shared_weights_info = {
            "word_embeddings": {
                "root": False,
                "sharing_with_ops": [0],
                "shape": [config.padded_vocab_size, config.hidden_size],
                "tp_split_dim": 0,
                "dp_split_dim": -1,
            }
        }
        self.output_extra_tensors_info = {}

    def forward(
        self,
        input_tensors: dict[str, Tensor],
        input_extra_tensors: dict[str, Tensor],
        output_extra_tensors: dict[str, Tensor],
    ):
        output_tensors = {}

        if type(input_tensors) is list:
            input_tensors = input_tensors[0]
        hidden_states: Tensor = input_tensors["hidden_states"]

        # Optional Layer norm post the cross-attention.
        # final_layernorm_output = self.final_layernorm(hidden_states)

        # always post process
        weights = self.embedding.word_embeddings.weight

        output, _ = self.output_layer(hidden_states, weights)

        labels = input_extra_tensors["labels"]

        if labels is None:
            output_tensors["output_tensor"] = output.transpose(0, 1).contiguous()
            return output_tensors
        else:
            labels = labels.transpose(0, 1).contiguous()
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
            else:
                loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
            loss = loss.transpose(0, 1).contiguous()

        output_tensors["output"] = loss

        return output_tensors
    

class OpLocalLayerNorm(OpModule):
    def __init__(self,
                 op_type: OpType,
                 op_name: str, 
                 config: TransformerConfig):
        super().__init__(op_type, op_name, config)
        
        self.layer_norm = FusedLayerNorm(self.config, self.config.hidden_size, self.config.layernorm_epsilon)
        
        self.input_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.output_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.weight_size = 0
        self.input_extra_tensors_info = {}
        self.output_extra_tensors_info = {}
    def forward(self, 
            input_tensors: dict[str, Tensor],
        input_extra_tensors: dict[str, Tensor],
        output_extra_tensors: dict[str, Tensor],
                ):
        output_tensors = {}
        hidden_states = input_tensors["hidden_states"]
        output = self.layer_norm(hidden_states)
        output_tensors["hidden_states"] = output
        return output_tensors


class OpTELayerNormPostProcess(OpModule):
    def __init__(
        self,
        op_type: OpType,
        op_name: str,
        config: TransformerConfig, 
        parallel_output: bool = True,
        num_tokentypes: int = 0,
        fp16_lm_cross_entropy: bool = False
    ):
        super().__init__(op_type, op_name, config)

        self.hidden_dropout = self.config.hidden_dropout

        self.final_layernorm = TENorm(self.config, config.hidden_size, config.layernorm_epsilon)
        if config.defer_embedding_wgrad_compute:
            self.embedding_activation_buffer = []
            self.grad_output_buffer = []
        else:
            self.embedding_activation_buffer = None
            self.grad_output_buffer = None

        self.parallel_output = parallel_output
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy

        self.output_layer = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            config.padded_vocab_size,
            config=config,
            init_method=self.config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not self.parallel_output,
            skip_weight_param_allocation=True,
            embedding_activation_buffer=self.embedding_activation_buffer,
            grad_output_buffer=self.grad_output_buffer,
        )

        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=config.padded_vocab_size,
            max_sequence_length=config.max_position_embeddings,
            position_embedding_type='learned_absolute',
            num_tokentypes=num_tokentypes,
        )


        self.weight_size = config.padded_vocab_size * config.hidden_size / self.tp_size

        self.input_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.output_tensors_info = {
            "output_tensor": {"shape": [1], "tp_split_dim": -1, "dp_split_dim": -1}
        }
        self.input_extra_tensors_info = {
            "labels": {
                "shape": [
                    config.micro_batch_size // self.dp_size,
                    config.seq_length
                ],
                "tp_split_dim": -1,
                "dp_split_dim": 0,
                "recv_from": 0,
            }
        }
        self.shared_weights_info = {
            "word_embeddings": {
                "root": False,
                "sharing_with_ops": [0],
                "shape": [config.padded_vocab_size, config.hidden_size],
                "tp_split_dim": 0,
                "dp_split_dim": -1,
            }
        }
        self.output_extra_tensors_info = {}

    def forward(
        self,
        input_tensors: dict[str, Tensor],
        input_extra_tensors: dict[str, Tensor],
        output_extra_tensors: dict[str, Tensor],
    ):
        output_tensors = {}

        if type(input_tensors) is list:
            input_tensors = input_tensors[0]
        hidden_states: Tensor = input_tensors["hidden_states"]

        # Optional Layer norm post the cross-attention.
        final_layernorm_output = self.final_layernorm(hidden_states)

        # always post process
        weights = self.embedding.word_embeddings.weight

        output, _ = self.output_layer(final_layernorm_output, weights)

        labels = input_extra_tensors["labels"]

        if labels is None:
            output_tensors["output_tensor"] = output.transpose(0, 1).contiguous()
            return output_tensors
        else:
            labels = labels.transpose(0, 1).contiguous()
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = tensor_parallel.vocab_parallel_cross_entropy(output, labels)
            else:
                loss = tensor_parallel.vocab_parallel_cross_entropy(output.float(), labels)
            loss = loss.transpose(0, 1).contiguous()

        output_tensors["output"] = loss

        return output_tensors
    


class OpTELayerNorm(OpModule):
    def __init__(self,
                 op_type: OpType,
                 op_name: str, 
                 config: TransformerConfig):
        super().__init__(op_type, op_name, config)
        
        self.layer_norm = TENorm(self.config, self.config.hidden_size, self.config.layernorm_epsilon)
        
        self.input_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.output_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.weight_size = 0
        self.input_extra_tensors_info = {}
        self.output_extra_tensors_info = {}
    def forward(self, 
            input_tensors: dict[str, Tensor],
        input_extra_tensors: dict[str, Tensor],
        output_extra_tensors: dict[str, Tensor],
                ):
        output_tensors = {}
        hidden_states = input_tensors["hidden_states"]
        output = self.layer_norm(hidden_states)
        output_tensors["hidden_states"] = output
        return output_tensors
    

class OpTELayerNormMlpDropout(OpModule):
    def __init__(
        self,
        op_type: OpType,
        op_name: str,
        config: TransformerConfig,
    ):
        super().__init__(op_type, op_name, config)

        self.layer_nummber = 1 
        self.hidden_dropout = config.hidden_dropout 
        

        self.pre_mlp_layernorm = IdentityOp()
        # TODO how to set the gpt_layer_spec.py when we have moe_frequency > 1,
        #      where MLP and MoE layer both appear alternately?
        self.mlp = MLP(config, MLPSubmodules(
                linear_fc1=TELayerNormColumnParallelLinear,
                linear_fc2=TERowParallelLinear,
            ))
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(self.layer_number)

        ## [Module 9: BiasDropoutFusion]
        self.mlp_bda = get_bias_dropout_add

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

        gemm_1_weight = config.hidden_size * config.ffn_hidden_size / self.tp_size
        gemm_2_weight = config.ffn_hidden_size * config.hidden_size / self.tp_size
        self.weight_size = gemm_1_weight + gemm_2_weight

        self.input_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.output_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.input_extra_tensors_info = {}
        self.output_extra_tensors_info = {}

    def forward(
        self,
        input_tensors: dict[str, Tensor],
        input_extra_tensors: dict[str, Tensor],
        output_extra_tensors: dict[str, Tensor],
    ):
        output_tensors = {}
        if type(input_tensors) is list:
            input_tensors = input_tensors[0]
        hidden_states: Tensor = input_tensors["hidden_states"]
        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        # TODO: could we move `bias_dropout_add_exec_handler` itself
        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(
                self.training, self.config.bias_dropout_fusion
            )(mlp_output_with_bias, residual, self.hidden_dropout)

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states,
            requires_grad=hidden_states.requires_grad,
            keep_graph=True,
        )
        output_tensors["hidden_states"] = output
        return output_tensors

class OpTELayerNormSelfAttentionDropout(OpModule):
    """
    input_layernorm + self_attention + self_attn_bda.

    """

    def __init__(
        self,
        op_type: OpType,
        op_name: str,
        config: TransformerConfig,
    ):
        """
        Args:
            layer_number: The global number of transformer layer, start with 1.
        """
        super().__init__(op_type, op_name, config)
        self.layer_number = 1
        self.hidden_dropout = config.hidden_dropout

        self.input_layernorm = IdentityOp() 
        self.self_attention = SelfAttention(self.config, SelfAttentionSubmodules(
                    linear_qkv=TELayerNormColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=IdentityOp,
                    k_layernorm=IdentityOp,
                ), 1, AttnMaskType.causal)

        self.self_attn_bda = get_bias_dropout_add

        # @jcasper how should we handle nvfuser?
        # Set bias+dropout+add fusion grad_enable execution handler.
        # TORCH_MAJOR = int(torch.__version__.split('.')[0])
        # TORCH_MINOR = int(torch.__version__.split('.')[1])
        # use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        # self.bias_dropout_add_exec_handler = nullcontext if use_nvfuser else torch.enable_grad
        self.bias_dropout_add_exec_handler = torch.enable_grad

        qkv_projection_size = config.kv_channels * config.num_attention_heads
        qkv_weight = config.hidden_size * qkv_projection_size * 3 / self.tp_size
        dense_weight = (
            (config.kv_channels * config.num_attention_heads) * config.hidden_size
        ) / self.tp_size
        self.weight_size = qkv_weight + dense_weight
        self.input_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.output_tensors_info = {
            "hidden_states": {
                "shape": self.hidden_state_size,
                "tp_split_dim": -1 if not config.sequence_parallel else 0,
                "dp_split_dim": 1,
            }
        }
        self.input_extra_tensors_info = {
            "attention_mask": {
                "shape": [
                    config.micro_batch_size // self.dp_size,
                    1,
                    config.seq_length,
                    config.seq_length,
                ],
                "tp_split_dim": -1,
                "dp_split_dim": -1,
                "recv_from": 0,
            }
        }
        self.output_extra_tensors_info = {}
    def forward(
        self,
        input_tensors: dict[str, Tensor],
        input_extra_tensors: dict[str, Tensor],
        output_extra_tensors: dict[str, Tensor],
    ):
        output_tensors = {}
        if type(input_tensors) is list:
            input_tensors = input_tensors[0]
        hidden_states: Tensor = input_tensors["hidden_states"]

        rotary_pos_emb = input_tensors.get("rotary_pos_emb", None)
        inference_params = input_tensors.get("inference_params", None)
        packed_seq_params = input_tensors.get("packed_seq_params", None)

        attention_mask: Tensor = input_extra_tensors["attention_mask"]
        # hidden_states: [s, b, h]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(
                self.training, self.config.bias_dropout_fusion
            )(attention_output_with_bias, residual, self.hidden_dropout)

        output_tensors["hidden_states"] = hidden_states
        return output_tensors