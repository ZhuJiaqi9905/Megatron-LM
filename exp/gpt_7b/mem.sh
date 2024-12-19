#!/bin/bash


TP=1
PP=1


export CUDA_DEVICE_MAX_CONNECTIONS=1
GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=7010
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# fixed Model related configuration here, pls not overlap with json config
HIDDEN_SIZE=4096
NUM_ATTENTION_HEADS=32
NUM_LAYERS=32
SEQ_LENGTH=$((4*1024))
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=${MICRO_BATCH_SIZE}


# GLOBAL_BATCH_SIZE=$((128*1024/${SEQ_LENGTH}))


VOCAB_FILE=/workspace/file/vocabs/gpt2-vocab.json
MERGE_FILE=/workspace/file/vocabs/gpt2-merges.txt


DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --mock-data \
"

# Model related configuration here, pls not overlap with json config
GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --num-layers ${NUM_LAYERS} \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 0.15 \
    --train-iters 500 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --tokenizer-type GPT2BPETokenizer \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --sequence-parallel \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --timing-log-level 2 \
    --timing-log-option all \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
"



export USE_FLASH_ATTN=1 && \
torchrun $DISTRIBUTED_ARGS \
    report_theoretical_memory.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \