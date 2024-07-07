#! /bin/bash

model=gpt3_350M

DATA_PATH=/mnt/gpu-91/dataset/gpt-dataset-simplewiki/my-gpt2_text_document
VOCAB_FILE=/mnt/gpu-91/dataset/gpt2-vocab.json
MERGE_FILE=/mnt/gpu-91/dataset/gpt2-merges.txt
CHECKPOINT_PATH=/mnt/gpu-91/varuna/checkpoints/${model}

if [[ "${model}" == "gpt3_350M" ]]; then
       NUM_LAYERS=24
       HIDDEN_SIZE=1024
       NUM_ATTENTION_HEADS=16
elif [[ "${model}" == "gpt3_1_3B" ]]; then
       NUM_LAYERS=24
       HIDDEN_SIZE=2048
       NUM_ATTENTION_HEADS=32
elif [[ "${model}" == "gpt3_2_7B" ]]; then
       NUM_LAYERS=32
       HIDDEN_SIZE=2560
       NUM_ATTENTION_HEADS=32
elif [[ "${model}" == "gpt3_6_7B" ]]; then
       NUM_LAYERS=32
       HIDDEN_SIZE=4096
       NUM_ATTENTION_HEADS=32
else
       echo "Don't have model ${model}"
       exit -1
fi

python scripts/rm_tmp.py
mkdir -p /mnt/gpu-91/varuna/profiles
chmod 777 /mnt/gpu-91/varuna/profiles
rm /mnt/gpu-91/varuna/profiles/*

# NCCL_SOCKET_IFNAME=enp NCCL_DEBUG=INFO GLOO_SOCKET_IFNAME=enp216s0np0,enp94s0np0

export GLOO_SOCKET_IFNAME=enp216s0np0 && \
python -m varuna.run_varuna --nstages 1 --chunk_size 1 --batch_size 1024 \
        --manager_ip 172.21.0.91 \
        --gpus_per_node 1 --no_morphing pretrain_gpt2.py \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTENTION_HEADS} \
        --seq-length 1024 \
        --max-position-embeddings 1024 \
        --train-iters 100 \
        --lr-decay-iters 100 \
        --data-path $DATA_PATH \
        --distributed-backend gloo \
        --vocab-file ${VOCAB_FILE} \
        --merge-file ${MERGE_FILE} \
        --save /mnt/gpu-91/varuna/profiles \
        --save-interval 1000 \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend gloo \
        --lr 0.00015 \
        --min-lr 1.0e-5 \
        --lr-decay-style cosine \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --warmup .01 \
        --use-cpu-initialization \
        --varuna --fp16 --fp16-lm-cross-entropy \
        --profiling

