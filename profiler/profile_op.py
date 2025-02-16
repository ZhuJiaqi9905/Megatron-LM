import time 
from megatron.training import initialize_megatron, get_args
from megatron.training.arguments import core_transformer_config_from_args
from model_configs import model_prof_configs, gpt_configs
import torch
from megatron.core.transformer.transformer_config import TransformerConfig
import copy
import traceback
import gc 
import torch.distributed as dist 
import numpy as np 
from torch import Tensor
from op import OpLanguageModelEmbedding, OpLocalLayerNorm, OpLocalLayerNormMlpDropout, OpLocalLayerNormPostProcess, OpLocalLayerNormSelfAttentionDropout, OpType, OpPostProcess,OpTELayerNorm,OpTELayerNormMlpDropout, OpTELayerNormPostProcess, OpTELayerNormSelfAttentionDropout, OpTECoreAttention
from megatron.core.transformer.module import Float16Module
from megatron.core import mpu
from megatron.core.utils import unwrap_model
import pandas as pd 
from datetime import datetime
from megatron.core.distributed import DistributedDataParallel as DDP

DATA_BASE = 4/(1024 * 1024)


# 初始化 DataFrame，定义列名
columns = [
    "model_name", "model_size" , "micro_batch_size", "sequence_length_q", "sequence_length_kv","fwd_compute", "bwd_compute", 
    "fwd_allocated", "fwd_reserved", "bwd_allocated", "bwd_reserved", 
    "input", "output", "weight"
]
data = pd.DataFrame(columns=columns)

def add_data(df, model_name, model_size, op, micro_batch_size, sequence_length_q,sequence_length_kv, fwd_compute, bwd_compute, 
             fwd_allocated, fwd_reserved, bwd_allocated, bwd_reserved, 
             input_size, output_size, weight_size):
    new_row = {
        "model_name": model_name,
        "model_size": model_size,
        "op": op,
        "micro_batch_size": micro_batch_size,
        "sequence_length_q": sequence_length_q,
        "sequence_length_kv": sequence_length_kv,
        "fwd_compute": fwd_compute,
        "bwd_compute": bwd_compute,
        "fwd_allocated": fwd_allocated,
        "fwd_reserved": fwd_reserved,
        "bwd_allocated": bwd_allocated,
        "bwd_reserved": bwd_reserved,
        "input": input_size,
        "output": output_size,
        "weight": weight_size,
    }
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

def print_rank0(str):
    if torch.distributed.get_rank() == 0:
        print(str)


def get_config(model_name: str, model_size: str)-> TransformerConfig:
    if model_name == "gpt":
        (
            num_layers,
            hidden_size,
            ffn_hidden_size,
            num_attention_heads,
            kv_channels,
            vocab_size,
            params_dtype,
        ) = gpt_configs[model_size]     
        set_params_dtype(params_dtype)
        args.num_layers = num_layers
        args.hidden_size = hidden_size
        args.ffn_hidden_size = ffn_hidden_size
        args.num_attention_heads = num_attention_heads
        args.kv_channels = kv_channels
        args.padded_vocab_size = vocab_size
        config = core_transformer_config_from_args(args)
        return config
    else:
        raise RuntimeError("Not Implement")

def set_params_dtype(params_dtype: str):
    global DATA_BASE
    args = get_args()

    if params_dtype == "fp32":
        DATA_BASE = 4 / (1024 * 1024)
        params_dtype = torch.float
        args.fp16 = False
        args.params_dtype = params_dtype
    elif params_dtype == "fp16":
        DATA_BASE = 2 / (1024 * 1024)
        params_dtype = torch.half
        args.fp16 = True
        args.params_dtype = params_dtype
    else:
        raise RuntimeError(f"data type {params_dtype} not supported.")


def get_op(op_name: str, config: TransformerConfig):
    op = None
    if op_name == "LanguageModelEmbedding":
        op = OpLanguageModelEmbedding(OpType.LanguageModelEmbedding, "LanguageModelEmbedding", config)
    elif op_name == "LocalLayerNormSelfAttentionDropout":
        op = OpLocalLayerNormSelfAttentionDropout(OpType.LocalLayerNormSelfAttentionDropout, "LocalLayerNormSelfAttentionDropout", config)
    elif op_name == "LocalLayerNormMlpDropout":
        op = OpLocalLayerNormMlpDropout(OpType.LocalLayerNormMlpDropout, "LocalLayerNormMlpDropout", config)
    elif op_name == "PostProcess":
        op = OpPostProcess(OpType.PostProcess, "PostProcess", config)
    elif op_name == "LocalLayerNorm":
        op = OpLocalLayerNorm(OpType.LocalLayerNorm, "LocalLayerNorm", config)
    elif op_name == "LocalLayerNormPostProcess":
        op = OpLocalLayerNormPostProcess(OpType.LocalLayerNormPostProcess, "LocalLayerNormPostProcess", config)
    elif op_name == "TELayerNormPostProcess":
        op = OpTELayerNormPostProcess(OpType.TELayerNormPostProcess, "TELayerNormPostProcess", config)
    elif op_name == "TELayerNormSelfAttentionDropout":
        op = OpTELayerNormSelfAttentionDropout(OpType.TELayerNormSelfAttentionDropout, "TELayerNormSelfAttentionDropout", config)
    elif op_name == "TELayerNorm":
        op = OpTELayerNorm(OpType.TELayerNorm, "TELayerNorm", config)
    elif op_name == "TELayerNormMlpDropout":
        op = OpTELayerNormMlpDropout(OpType.TELayerNormMlpDropout, "TELayerNormMlpDropout", config)
    elif op_name == "TECoreAttention":
        op = OpTECoreAttention(OpType.TECoreAttention, "TECoreAttention", config)
    op.cuda(torch.cuda.current_device())
    args = get_args()
    if args.fp16:
        op = Float16Module(config, op) 
    op = DDP(config,
            op,
            data_parallel_group=mpu.get_data_parallel_group(with_context_parallel=True),
            expert_data_parallel_group=mpu.get_data_modulo_expert_parallel_group(),
            accumulate_allreduce_grads_in_fp32=args.accumulate_allreduce_grads_in_fp32,
            overlap_grad_reduce=args.overlap_grad_reduce,
            use_distributed_optimizer=args.use_distributed_optimizer,
            # Turn off bucketing for model_chunk 2 onwards, since communication for these
            # model chunks is overlapped with compute anyway.
            disable_bucketing=True,
            check_for_nan_in_grad=args.check_for_nan_in_loss_and_grad)
    return op

def profile_op_compute(op, op_name, input_tensors, input_extra_tensors, input_tensors_for_bwd, output_extra_tensors):
    grad_type = torch.float
    args = get_args()


    if op_name in ["PostProcess", "LocalLayerNormPostProcess", "TELayerNormSelfAttentionDropout", "TELayerNormMlpDropout", "TELayerNorm","TELayerNormPostProcess"]: # Those op need to do "one fwd, one bwd"
        ## warm-up
        for _ in range(args.prof_warmup_times):
            # forward
            output_tensors = op(input_tensors, input_extra_tensors, output_extra_tensors)  
            # backward
            origin_outputs, output_grads = get_outputs_and_grads(output_tensors, output_extra_tensors, grad_type) 
            if op_name in ["LanguageModelEmbedding"]: # embedding is the first operator, we do not need to compute the grad of the input
                torch.autograd.backward(origin_outputs, grad_tensors=output_grads, retain_graph=True)
            else: # for other operators, we need to compute the grad of both the input and the weight 
                torch.autograd.grad(outputs=origin_outputs, grad_outputs=output_grads, inputs=input_tensors_for_bwd, allow_unused=False, retain_graph=True)
        ## profile 
        sum_fwd_time = 0
        sum_bwd_time = 0
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.prof_repeat_times[0]):
            # forward
            output_tensors = op(input_tensors, input_extra_tensors, output_extra_tensors) 
        end.record() 
        torch.cuda.synchronize()
        sum_fwd_time += start.elapsed_time(end)
        avg_fwd_time = sum_fwd_time * 1000 / args.prof_repeat_times[0]
         
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)   
        start.record()     
        for _ in range(args.prof_repeat_times[0]):
            # forward
            output_tensors = op(input_tensors, input_extra_tensors, output_extra_tensors) 
            # backward
            origin_outputs, output_grads = get_outputs_and_grads(output_tensors, output_extra_tensors, grad_type) 
            if op_name in ["LanguageModelEmbedding"]: # embedding is the first operator, we do not need to compute the grad of the input
                torch.autograd.backward(origin_outputs, grad_tensors=output_grads, retain_graph=True)
            else: # for other operators, we need to compute the grad of both the input and the weight 
                torch.autograd.grad(outputs=origin_outputs, grad_outputs=output_grads, inputs=input_tensors_for_bwd, allow_unused=False, retain_graph=True) 
        end.record()
        torch.cuda.synchronize()
        sum_bwd_time = start.elapsed_time(end)  - sum_fwd_time
        avg_bwd_time = sum_bwd_time * 1000 / args.prof_repeat_times[0]
        return avg_fwd_time, avg_bwd_time

    # Other ops
    ## forward warm-up
    for _ in range(args.prof_warmup_times):
        output_tensors = op(input_tensors, input_extra_tensors, output_extra_tensors) 

    ##### forward, sync after all runs
    sum_fwd_time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.prof_repeat_times[0]):
        output_tensors = op(input_tensors, input_extra_tensors, output_extra_tensors)          
    end.record()
    torch.cuda.synchronize()
    
    sum_fwd_time += start.elapsed_time(end)          
    avg_fwd_time = sum_fwd_time * 1000 / args.prof_repeat_times[0]
    
    # backward warm-up
    origin_outputs, output_grads = get_outputs_and_grads(output_tensors, output_extra_tensors, grad_type) 
    sum_bwd_time = 0
    ### warmup for backward
    for _i in range(args.prof_warmup_times):
        if op_name in ["LanguageModelEmbedding"]: # embedding is the first operator, we do not need to compute the grad of the input
            torch.autograd.backward(origin_outputs, grad_tensors=output_grads, retain_graph=True)
        else: # for other operators, we need to compute the grad of both the input and the weight 
            torch.autograd.grad(outputs=origin_outputs, grad_outputs=output_grads, inputs=input_tensors_for_bwd, allow_unused=False, retain_graph=True)

    ## backward, sync after all run
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record() 
    for _ in range(args.prof_repeat_times[0]):
        if op_name in ["LanguageModelEmbedding"]:
            torch.autograd.backward(origin_outputs, grad_tensors=output_grads, retain_graph=True)
        else:
            torch.autograd.grad(outputs=origin_outputs, grad_outputs=output_grads, inputs=input_tensors_for_bwd, allow_unused=False, retain_graph=True)
    end.record()
    torch.cuda.synchronize()
    sum_bwd_time += start.elapsed_time(end) 
    avg_bwd_time = sum_bwd_time * 1000 / args.prof_repeat_times[0]
    
    return avg_fwd_time, avg_bwd_time



def profile_op_memory(op, op_name, input_tensors, input_extra_tensors, input_tensors_for_bwd, output_extra_tensors):
    ## Profiling memory
    grad_type = torch.float
    _mem_reserved_fwd = 0 
    _mem_reserved_bwd = 0 
    _mem_allocated = 0

    output_data = None
    origin_outputs = None
    output_grads = None
    torch.cuda.empty_cache()



    mem_allocated = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    mem_reserved = torch.cuda.memory_reserved() / (1024.0 * 1024.0)

    output_data = op(input_tensors, input_extra_tensors, output_extra_tensors)

    new_mem_allocated = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    new_mem_reserved = torch.cuda.memory_reserved() / (1024.0 * 1024.0)

    _mem_reserved_fwd = new_mem_reserved - mem_reserved
    _mem_allocated = new_mem_allocated - mem_allocated
    _mem_reserved_fwd -= _mem_allocated
    if _mem_reserved_fwd < 0:
        _mem_reserved_fwd = 0

    outputs = []
    output_grads = []
    for output_name in output_data:
        outputs.append(output_data[output_name])
        output_grads.append(torch.randn(output_data[output_name].size(), requires_grad=False, device=torch.cuda.current_device(), dtype=grad_type) )
    for output_extra_name in output_extra_tensors:
        outputs.append(output_extra_tensors[output_extra_name])
        output_grads.append(torch.randn(output_extra_tensors[output_extra_name].size(), requires_grad=False, device=torch.cuda.current_device(), dtype=grad_type) )

    if op_name in ["LanguageModelEmbedding"]:       
        torch.autograd.backward(outputs, grad_tensors=output_grads, retain_graph=True)                    
    else:
        input_tensors_for_bwd = []
        for input_name in input_tensors:
            input_tensors_for_bwd.append(input_tensors[input_name])
        for input_extra_name in input_extra_tensors:
            ## workaround for softmax op.
            if "mask" not in input_extra_name and "bias" not in input_extra_name  and "labels" not in input_extra_name:
                input_tensors_for_bwd.append(input_extra_tensors[input_extra_name])
        torch.autograd.grad(outputs=outputs, grad_outputs=output_grads, inputs=input_tensors_for_bwd, allow_unused=False, retain_graph=True)

    mem_allocated_bwd = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
    mem_reserved_bwd = torch.cuda.memory_reserved() / (1024.0 * 1024.0)

    _mem_reserved_bwd = mem_reserved_bwd - new_mem_reserved
    _mem_allocated_bwd = mem_allocated_bwd - new_mem_allocated
    if _mem_reserved_bwd < 0:
        _mem_reserved_bwd = 0
    
    return _mem_allocated, _mem_reserved_fwd, _mem_allocated_bwd, _mem_reserved_bwd 
    
    
def profile_op(op_name: str, config: TransformerConfig):
    
    op = get_op(op_name, config) 
    args = get_args()
    sum_input_size, sum_output_size, weight_size, input_shape_dict, input_extra_dict = infer_data_size(op) 
    
    input_tensors, input_extra_tensors = get_inputs(input_shape_dict, input_extra_dict, args.params_dtype, config)
    input_tensors_for_bwd = get_input_tensors_for_bwd(op_name, input_tensors, input_extra_tensors)
    output_extra_tensors = {}

    fwd_allocated, fwd_reserved, bwd_allocated, bwd_reserved = profile_op_memory(op, op_name, input_tensors, input_extra_tensors, input_tensors_for_bwd, output_extra_tensors)
    fwd_time, bwd_time = profile_op_compute(op,op_name, input_tensors, input_extra_tensors, input_tensors_for_bwd, output_extra_tensors)
    return fwd_time, bwd_time, fwd_allocated, fwd_reserved, bwd_allocated, bwd_reserved, sum_input_size, sum_output_size, weight_size
    

    
    

def infer_data_size(op):
    '''
    Infer each op's input/output tensor shape, which will be used to generate input/output tensor during the profiling.
    '''
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    prev_extra_size = 0

    op = unwrap_model(op, (DDP, Float16Module))
    
    weight_size = np.prod(op.weight_size) * DATA_BASE

    sum_input_size = 0
    sum_output_size = 0
    input_shape_dict = {}
    input_extra_dict = {}
    import pdb 
    # if torch.distributed.get_rank() == 0:
    #     pdb.set_trace()
    ## infer input tensor size
    for input_name in op.input_tensors_info:
        input_shape = copy.deepcopy(op.input_tensors_info[input_name]["shape"])
        tp_split_dim = op.input_tensors_info[input_name]["tp_split_dim"] 
        cp_split_dim = op.input_tensors_info[input_name]["cp_split_dim"]
        if tp_split_dim != -1:
            input_shape[tp_split_dim] = input_shape[tp_split_dim] // tp_size
        if cp_split_dim != -1:
            input_shape[cp_split_dim] = input_shape[cp_split_dim] // cp_size
        input_shape_dict[input_name] = input_shape
        sum_input_size += np.prod(input_shape) * DATA_BASE
    sum_input_size += prev_extra_size

    ## infer output tensor size
    for output_name in op.output_tensors_info:
        output_shape = copy.deepcopy(op.output_tensors_info[output_name]["shape"])
        tp_split_dim = op.output_tensors_info[output_name]["tp_split_dim"] 
        cp_split_dim = op.output_tensors_info[output_name]["cp_split_dim"]
        if tp_split_dim != -1:
            output_shape[tp_split_dim] = output_shape[tp_split_dim] // tp_size
        if cp_split_dim != -1:
            output_shape[cp_split_dim] = output_shape[cp_split_dim] // cp_size
        sum_output_size += np.prod(output_shape) * DATA_BASE

    ## infer input extra tensor size
    sum_input_extra_size = 0

    for input_extra_name in op.input_extra_tensors_info:
        input_shape = copy.deepcopy(op.input_extra_tensors_info[input_extra_name]["shape"])
        tp_split_dim = op.input_extra_tensors_info[input_extra_name]["tp_split_dim"] 
        cp_split_dim = op.input_extra_tensors_info[input_extra_name]["cp_split_dim"]
        if tp_split_dim != -1:
            input_shape[tp_split_dim] = input_shape[tp_split_dim] // tp_size
        if cp_split_dim != -1:
            input_shape[cp_split_dim] = input_shape[cp_split_dim] // cp_size
        input_extra_dict[input_extra_name] = input_shape
        ## current workaround for masks.
        if "mask" not in input_extra_name and "label" not in input_extra_name: # we do not need to transfer attention_mask to other stage
            sum_input_extra_size += np.prod(input_shape) * DATA_BASE

    ## infer output extra tensor size
    sum_output_extra_size = 0
    for output_extra_name in op.output_extra_tensors_info:
        output_shape = copy.deepcopy(op.output_extra_tensors_info[output_extra_name]["shape"])
        tp_split_dim = op.output_extra_tensors_info[output_extra_name]["tp_split_dim"] 
        if tp_split_dim != -1:
            output_shape[tp_split_dim] = output_shape[tp_split_dim] // tp_size
        if cp_split_dim != -1:
            output_shape[cp_split_dim] = output_shape[cp_split_dim] // cp_size
        sum_output_extra_size += np.prod(output_shape) * DATA_BASE

    current_extra_size = prev_extra_size + sum_output_extra_size - sum_input_extra_size
    # print(f"{op.op_name}: current_extra_size = {current_extra_size}= prev_extra_size {prev_extra_size}+ sum_output_extra_size {sum_output_extra_size}- sum_input_extra_size {sum_input_extra_size}")
    sum_output_size += current_extra_size
    prev_extra_size = current_extra_size

    # torch.distributed.barrier()
    
    return sum_input_size, sum_output_size, weight_size, input_shape_dict, input_extra_dict

def get_input_tensors_for_bwd(op_name: str, input_tensors: dict, input_extra_tensors: dict):
    if op_name in ["LanguageModelEmbedding"]:
        input_tensors_for_bwd = None
    else:
        input_tensors_for_bwd = []
        for input_name in input_tensors:
            input_tensors_for_bwd.append(input_tensors[input_name])
        for input_extra_name in input_extra_tensors:
            ## workaround for softmax op.
            if "mask" not in input_extra_name and "bias" not in input_extra_name  and "labels" not in input_extra_name: 
                # Attention mask / labels not need grad.
                input_tensors_for_bwd.append(input_extra_tensors[input_extra_name])   

    return input_tensors_for_bwd

def get_outputs_and_grads(output_tensors: dict, output_extra_tensors, grad_type):
    '''
    get the outputs and grad_outputs for autograd backward.
    '''
    ## keep original output tensors 
    origin_outputs = []
    for output_name in output_tensors:
        if output_tensors[output_name] == None:
            continue
        origin_outputs.append(output_tensors[output_name])
    for output_extra_name in output_extra_tensors:
        if output_extra_tensors[output_extra_name] == None:
            continue
        origin_outputs.append(output_extra_tensors[output_extra_name])

    output_grads = []
    ## add one more dummy op for each output tensor, in order to get the size of the grad tensor of the output.
    for output_tensor in origin_outputs:
        if output_tensor == None:
            continue
        tensor_shape = list(output_tensor.size())
        if len(tensor_shape) >= 3:
            pool_op = torch.nn.AdaptiveMaxPool2d(1)
        elif len(tensor_shape) >= 2:
            pool_op = torch.nn.AdaptiveMaxPool1d(1)
        else:
            pool_op = torch.nn.Identity()
        output_tensor_ = pool_op(output_tensor)
        
        output_tensor_grad = torch.randn(output_tensor_.size(), requires_grad=False, device=torch.cuda.current_device(), dtype=grad_type)
        origin_grad = torch.autograd.grad(outputs=output_tensor_, grad_outputs=output_tensor_grad, inputs=output_tensor, allow_unused=False, retain_graph=False)
        output_grads.append(origin_grad[0])    

    return origin_outputs, output_grads


def get_inputs(input_shape_dict: dict, input_extra_dict: dict, params_dtype: str, config: TransformerConfig):
    inputs = {}
    input_extra_tensors = {}
    
    for input_name in input_shape_dict:
        input_shape = input_shape_dict[input_name]

        if input_name in ["input_ids", "position_ids"]:
            inputs[input_name] = torch.randint(0, 8192, input_shape, requires_grad=False, device=torch.cuda.current_device(), dtype=torch.long)
        else:
            inputs[input_name] = torch.rand(input_shape, requires_grad=True, device=torch.cuda.current_device(), dtype=params_dtype)        

    for input_extra_name in input_extra_dict:
        input_shape = input_extra_dict[input_extra_name]
        if input_extra_name in ["attention_mask"]:
            input_extra_tensors[input_extra_name] = get_attention_mask(input_shape) 
        elif input_extra_name in ["labels"]:
            input_shape = input_extra_dict[input_extra_name]
            input_extra_tensors[input_extra_name] = torch.rand(input_shape, requires_grad=False, device=torch.cuda.current_device()).long() * config.padded_vocab_size
        else:
            input_extra_tensors[input_extra_name] = torch.rand(input_shape, requires_grad=True, device=torch.cuda.current_device(), dtype=params_dtype)

    

    return inputs, input_extra_tensors

def get_attention_mask(input_shape: list[int]):
    input_shape = copy.deepcopy(input_shape)
    cp_size = mpu.get_context_parallel_world_size()
    seq_dim = 2
     
    input_shape[seq_dim] = input_shape[seq_dim] * cp_size
    # generate causal attention mask
    val = torch.tril(torch.ones(input_shape, device=torch.cuda.current_device())) 
    if cp_size > 1:
        cp_rank = mpu.get_context_parallel_rank()
        val = val.view(
            *val.shape[0:seq_dim],
            2 * cp_size,
            val.shape[seq_dim] // (2 * cp_size),
            *val.shape[(seq_dim + 1) :],
        )
        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], 
                                device="cpu", pin_memory=True).cuda(non_blocking=True)
        val = val.index_select(seq_dim, index)
        val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
    return val    


def run_profile(task):
    global data
    
    model_name = task["model_name"]
    model_size = task["model_size"]
    mbs_list = task["mbs_list"]
    seq_lens_list = task["seq_lens_list"]
    
    args = get_args()
    
    config = get_config(model_name, model_size)
    
    tp_size = mpu.get_tensor_model_parallel_world_size()

    if args.prof_core_attention:
        seq_len_kv_list = task["seq_len_kv_list"]
        assert seq_len_kv_list is not None, 'prof-core-attention need seq_len_kv_list' 
        op_name = "TECoreAttention"
        for mbs in mbs_list:
            for seq_len_kv in seq_len_kv_list:
                for seq_len in seq_lens_list:
                        new_config = copy.deepcopy(config)
                        new_config.micro_batch_size = mbs 
                        new_config.seq_length = seq_len
                        new_config.max_sequence_length = seq_len
                        new_config.seq_length_kv = seq_len_kv
                        print_rank0(f"{model_name}_{model_size}.start profile {op_name}. mbs = {mbs}, seq_len_q = {seq_len}, seq_len_kv = {seq_len_kv}, tp = {tp_size} ...")  
                        try: 
                            fwd_time, bwd_time, fwd_allocated, fwd_reserved, bwd_allocated, bwd_reserved, input_size, output_size, weight_size = profile_op(op_name, new_config)
                        except RuntimeError as e:
                            print(f"RuntimeError: {e}. {traceback.format_exc()}")
                            fwd_time, bwd_time, fwd_allocated, fwd_reserved, bwd_allocated, bwd_reserved, input_size, output_size, weight_size = 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000
                        print_rank0(f"[results] {op_name}: fwd_compute = {fwd_time:.2f} us, bwd_compute = {bwd_time:.2f} us, fwd_allocated = {fwd_allocated:.2f} MB, fwd_reserved = {fwd_reserved:.2f} MB, bwd_allocated = {bwd_allocated:.2f} MB, bwd_reserved = {bwd_reserved:.2f} MB. input = {input_size:.2f} MB. output = {output_size:.2f} MB. weight = {weight_size:.2f} MB")
                        data = add_data(data, model_name, model_size, op_name, mbs, seq_len, seq_len_kv, fwd_time, bwd_time, fwd_allocated, fwd_reserved, bwd_allocated, bwd_reserved, input_size, output_size, weight_size)
                        gc.collect()
                        dist.barrier()

        
    for op_name in model_prof_configs[model_name]["ops"]:
        for seq_len in seq_lens_list:
            for mbs in mbs_list:
                new_config = copy.deepcopy(config)
                new_config.micro_batch_size = mbs 
                new_config.seq_length = seq_len
                new_config.max_sequence_length = seq_len
            
                print_rank0(f"{model_name}_{model_size}.start profile {op_name}. mbs = {mbs}, seq_len = {seq_len}, tp = {tp_size} ...")  
                try: 
                    fwd_time, bwd_time, fwd_allocated, fwd_reserved, bwd_allocated, bwd_reserved, input_size, output_size, weight_size = profile_op(op_name, new_config)
                except RuntimeError as e:
                    print(f"RuntimeError: {e}. {traceback.format_exc()}")
                    fwd_time, bwd_time, fwd_allocated, fwd_reserved, bwd_allocated, bwd_reserved, input_size, output_size, weight_size = 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000, 10000000
                print_rank0(f"[results] {op_name}: fwd_compute = {fwd_time:.2f} us, bwd_compute = {bwd_time:.2f} us, fwd_allocated = {fwd_allocated:.2f} MB, fwd_reserved = {fwd_reserved:.2f} MB, bwd_allocated = {bwd_allocated:.2f} MB, bwd_reserved = {bwd_reserved:.2f} MB. input = {input_size:.2f} MB. output = {output_size:.2f} MB. weight = {weight_size:.2f} MB")
                data = add_data(data, model_name, model_size, op_name, mbs, seq_len, seq_len, fwd_time, bwd_time, fwd_allocated, fwd_reserved, bwd_allocated, bwd_reserved, input_size, output_size, weight_size)
                gc.collect()
                dist.barrier()




if __name__ == "__main__":
    
    start_profiling_time = time.time()
    
    initialize_megatron()
    args = get_args()
    model_names = ["resnet", "gpt", "t5"] if args.prof_model_name == "all" else [args.prof_model_name]
    all_prof_tasks = []
    
    for model_name in model_names:
        model_sizes = model_prof_configs[model_name]["model_size"] if args.prof_model_size == "all" else [args.prof_model_size]
        for model_size in model_sizes:
            if args.prof_mbs_list is None:
                micro_batch_sizes = model_prof_configs[model_name]["mbs"]
            else:
                micro_batch_sizes = args.prof_mbs_list
            if args.prof_seq_lens_list is None:
                seq_lens = model_prof_configs[model_name]["seq_len"]
            seq_len_q_list = None 
            seq_len_kv_list = None 
            if "seq_len_kv" in model_prof_configs[model_name]:
                seq_len_kv_list = model_prof_configs[model_name]["seq_len_kv"]
                
            all_prof_tasks.append({"model_name": model_name, "model_size": model_size, "mbs_list": micro_batch_sizes, "seq_lens_list": seq_lens,"seq_len_kv_list": seq_len_kv_list})
        ## run profiling tasks
    for prof_task in all_prof_tasks:
        run_profile(prof_task)
    end_profiling_time = time.time()
    if torch.distributed.get_rank() == 0: 
        data.to_csv(f"{'.' if args.prof_path is None else args.prof_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.prof_model_name}-{args.prof_model_size}_tp_{mpu.get_tensor_model_parallel_world_size()}_cp_{mpu.get_context_parallel_world_size()}.csv", index=False)
    print_rank0(f"[TOTAL PROFILING TIME] {end_profiling_time - start_profiling_time:2f} s")
