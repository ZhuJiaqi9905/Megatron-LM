#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_SOCKET_IFNAME=enp
export NCCL_DEBUG=INFO
GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=172.21.0.42
MASTER_PORT=6000
NNODES=4
NODE_RANK=${1}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/workspace/python/Megatron-LM/data/checkpoint/gpt3_350M/
VOCAB_FILE=/workspace/python/Megatron-LM/dataset/gpt2-vocab.json
MERGE_FILE=/workspace/python/Megatron-LM/dataset/gpt2-merges.txt
DATA_PATH=/workspace/python/Megatron-LM/dataset/gpt-dataset-simplewiki/my-gpt2_text_document


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --num-layers 24 \
    --hidden-size 1024 \
    --num-attention-heads 16 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --micro-batch-size 8 \
    --global-batch-size 1024 \
    --lr 0.00015 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --attention-softmax-in-fp32
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    "
    # --split 949,50,1
DATA_ARGS="--mock-data     --vocab-file $VOCAB_FILE "
OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10 \
    --eval-interval 1000 \
    --eval-iters 30
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    # --load $CHECKPOINT_PATH

