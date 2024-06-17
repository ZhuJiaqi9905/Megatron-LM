
DATA_PATH=/workspace/python/Megatron-LM/dataset/gpt-dataset-simplewiki/my-gpt2_text_document
VOCAB_FILE=/workspace/python/Megatron-LM/dataset/gpt2-vocab.json
MERGE_FILE=/workspace/python/Megatron-LM/dataset/gpt2-merges.txt
GPUS_PER_SERVER=4

# NCCL_SOCKET_IFNAME=enp NCCL_DEBUG=INFO GLOO_SOCKET_IFNAME=enp216s0np0,enp94s0np0

python -m varuna.run_varuna --batch_size 8192 \
        --manager_ip 172.21.0.91 \
        --gpus_per_node $GPUS_PER_SERVER --no_morphing pretrain_gpt2.py \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --train-iters 100 \
        --lr-decay-iters 100 \
        --data-path $DATA_PATH \
        --distributed-backend gloo \
        --vocab-file ${VOCAB_FILE} \
        --merge-file ${MERGE_FILE} \
        --save /workspace/python/Megatron-LM-varuna/profiles \
        --save-interval 1000 \
        --data-impl mmap \
        --split 949,50,1 \
        --lr 0.00001 \
        --min-lr 1e-5 \
        --lr-decay-style cosine \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --use-cpu-initialization \
        --warmup .05 \
        --fp16 \
        --varuna \
        --profiling

