#! /bin/bash
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
export PYTHONPATH="/workspace/python/Megatron-LM-0.6.0/:${PYTHONPATH}:/workspace/python/Megatron-LM/:/workspace/Megatron-LM/"
export CUDA_DEVICE_MAX_CONNECTIONS=1
MASTER_ADDR=localhost
MASTER_PORT=7010
NNODES=1
NODE_RANK=0

RUNTIME_PATH=$(pwd)/
PROFILING_PATH=${RUNTIME_PATH}/aws/a10g/

VOCAB_FILE=../vocabs/gpt2-vocab.json
MERGE_FILE=../vocabs/gpt2-merges.txt
#  num_layers, seq_len, hidden_size, ffn_hidden_size, num_attention_heads, kv_channels, vocab_size, params_dtype are fake.
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16
SEQ_LENGTH=2048
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16


DATA_ARGS="
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --mock-data \
"
GPT_ARGS="
    --num-layers 1 \
    --no-async-tensor-model-parallel-allreduce \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 0.00015 \
    --train-iters 20 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --tokenizer-type GPT2BPETokenizer \
    --use-mcore-models \
    --transformer-impl local \
    --sequence-parallel \
"
mkdir -p ${PROFILING_PATH}
MAX_NUM_GPUS=4
MODEL_NAME=gpt
MODEL_SIZE=1-3B

for ((tp_size=2; tp_size<=$MAX_NUM_GPUS; tp_size=tp_size*2))
do
    GPUS_PER_NODE=${tp_size}
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

    echo [TIME] before profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log

    torchrun $DISTRIBUTED_ARGS \
        profile_op.py \
        ${DATA_ARGS} \
        ${GPT_ARGS} \
        --tensor-model-parallel-size ${tp_size} \
        --use-mcore-models \
        --prof-path $PROFILING_PATH \
        --prof-model-name $MODEL_NAME \
        --prof-model-size $MODEL_SIZE \
        --prof-warmup-times 1 \
        --prof-repeat-times 5 \
        2>&1 | tee ${PROFILING_PATH}profiling_${MODEL_NAME}_${MODEL_SIZE}_op_tp${tp_size}.log

    echo [TIME] after profiling tp_size $tp_size : $(date '+%Y-%m-%d-%H-%M-%S') >> ${PROFILING_PATH}profiling_${MODEL_NAME}.log
done