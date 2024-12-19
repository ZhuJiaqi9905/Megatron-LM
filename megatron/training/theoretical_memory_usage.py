# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical memory footprint for model training."""


import math

NUM_BYTES_IN_MEGABYTE = 1024 * 1024


def compute_weight_and_optimizer_memory(args, verbose=False):
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts = 1 if args.num_experts is None else args.num_experts
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1
    num_parameters_in_transformer_layers = (
        2
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            1
            + ((args.ffn_hidden_size / args.hidden_size) * num_experts * gated_linear_multiplier)
            + (args.num_query_groups / args.num_attention_heads)
            + (2 / args.hidden_size)
            + (1 / (args.num_layers * args.hidden_size))
        )
    )
    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size
    num_total_parameters = num_parameters_in_transformer_layers + num_parameters_in_embedding_layers
    if verbose:
        print(
            f"Number of parameters in transformer layers in billions: {num_parameters_in_transformer_layers / 10**9: .2f}"
        )
        print(
            f"Number of parameters in embedding layers in billions: {num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")

    # Most loaded model shard has (1/pp_size transformer layers + 1 embedding layer) / tp_size.
    num_parameters_on_most_loaded_model_shard = (
        (num_parameters_in_transformer_layers / args.pipeline_model_parallel_size) + embedding_size
    ) / args.tensor_model_parallel_size
    if args.untie_embeddings_and_output_weights and args.pipeline_model_parallel_size == 1:
        num_parameters_on_most_loaded_model_shard += (
            embedding_size / args.tensor_model_parallel_size
        )
    if verbose:
        print(
            f"Number of parameters in most loaded shard in billions: {num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        )

    if args.pipeline_model_parallel_size > 1:
        # Other shards just have (1/pp_size transformer layers) / tp_size.
        num_parameters_on_other_model_shards = num_parameters_in_transformer_layers / (
            args.pipeline_model_parallel_size * args.tensor_model_parallel_size
        )
        if verbose:
            print(
                f"Number of parameters in other shards in billions: {num_parameters_on_other_model_shards / 10**9:.4f}"
            )

    num_bytes_per_parameter = (
        18 if not args.use_distributed_optimizer else 6 + (12 / args.data_parallel_size)
    )
    weight_and_optimizer_memory = (
        num_parameters_on_most_loaded_model_shard * num_bytes_per_parameter
    )

    return weight_and_optimizer_memory


def compute_activation_memory(args, num_microbatches, verbose=False):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this function
    # are for the first pipeline stage.

    # Memory footprint from transformer layer (self-attention and MLP).

    if args.sequence_parallel and (args.recompute_granularity == 'selective' or (args.transformer_impl == 'transformer_engine' and args.use_mcore_models)): # TP + selective recompute + sequence parallel
        activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
            18 + (4 * (args.ffn_hidden_size / args.hidden_size))
        ) / args.tensor_model_parallel_size
    elif not args.sequence_parallel and (args.recompute_granularity == 'selective' or  (args.transformer_impl == 'transformer_engine' and args.use_mcore_models)) : # TP + selective recompute
        activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
            10 + (8 + (4 * (args.ffn_hidden_size / args.hidden_size))) / args.tensor_model_parallel_size
        )
    elif not args.sequence_parallel and args.recompute_granularity == None: # TP
        activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
            10 + (8 + (4 * (args.ffn_hidden_size / args.hidden_size))) / args.tensor_model_parallel_size
        ) + 5 * args.num_attention_heads * args.seq_length * args.seq_length * args.micro_batch_size / args.tensor_model_parallel_size
    elif args.sequence_parallel and args.recompute_granularity == None: # TP + sequence parallel
        activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
            (18 + (4 * (args.ffn_hidden_size / args.hidden_size))) / args.tensor_model_parallel_size
        ) + 5 * args.num_attention_heads * args.seq_length * args.seq_length * args.micro_batch_size / args.tensor_model_parallel_size        
    else:
        raise RuntimeError("Not support this config")
    if verbose:
        print(
            f"Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_MEGABYTE:.1f} MB"
        )
    activation_memory *= args.num_layers # the number of layers of the first stage is (args.num_layers / args.pipeline_model_parallel_size)

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += (
        8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    ) # I think that the activation of embedding should not be splited by TP size.
     
    
    # Dropout in embedding layer (pp_size microbatches in flight).
    if args.sequence_parallel: 
        activation_memory += (
            args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            * args.pipeline_model_parallel_size
        ) / args.tensor_model_parallel_size
    else:
        activation_memory += (
            args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            * args.pipeline_model_parallel_size
        )      
    # Multiply by interleaved PP memory factor.
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if args.pipeline_model_parallel_size == 1:
        # Inputs to output layer and CE loss.
        if args.sequence_parallel:
            activation_memory += (
                args.seq_length
                * args.micro_batch_size
                * args.hidden_size
                * 4
                * (1 + (args.padded_vocab_size / args.hidden_size))
            ) / args.tensor_model_parallel_size
        else:
            activation_memory += (
                args.seq_length
                * args.micro_batch_size
                * args.hidden_size
                * 4
                * (args.padded_vocab_size / args.hidden_size)
            ) / args.tensor_model_parallel_size # cross entropy
            + ( args.seq_length # layernorm + output layer projection
                * args.micro_batch_size
                * args.hidden_size
                * 4)
            
            
    # Activation memory is partitioned by TP size due to tensor and sequence model parallelism.
    return activation_memory 


def report_theoretical_memory(args, num_microbatches=None, verbose=False):
    # Formulae here assume sequence parallelism and selective activation recomputation.
    # if not args.sequence_parallel or args.recompute_granularity != 'selective':
    #     print(f"sequence_parallel: {args.sequence_parallel}. recompute_granularity: {args.recompute_granularity}")
    #     return

    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose) / NUM_BYTES_IN_MEGABYTE
    )
    activation_memory = (
        compute_activation_memory(args, num_microbatches=num_microbatches, verbose=verbose)
        / NUM_BYTES_IN_MEGABYTE
    )
    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB, "
        f"activation={activation_memory:.2f} MB, "
        f"total={total_memory:.2f} MB\n"
    )
