#! /bin/bash

model=gpt3_350M
nstages=8
mbs=1

gbs=1024
gpus_per_node=1
total_gpus=16


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

DATA_PATH=/mnt/gpu-91/dataset/gpt-dataset-simplewiki/my-gpt2_text_document
VOCAB_FILE=/mnt/gpu-91/dataset/gpt2-vocab.json
MERGE_FILE=/mnt/gpu-91/dataset/gpt2-merges.txt
CHECKPOINT_PATH=/mnt/gpu-91/varuna/checkpoints/${model}

# rm _tmp_*
rm -rf ${CHECKPOINT_PATH}/*

export GLOO_SOCKET_IFNAME=enp216s0np0,enp94s0np0 && \
python3 -m varuna.run_varuna \
       --nstages 8 --chunk_size 2 \
       --manager_ip 172.21.0.91 \
       --batch_size ${gbs} \
       --total_gpus ${total_gpus} \
       --gpus_per_node ${gpus_per_node} \
       --no_morphing pretrain_gpt2.py \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_ATTENTION_HEADS \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --train-iters 18750 \
       --lr-decay-iters 18750 \
       --save ${CHECKPOINT_PATH} \
       --data-path ${DATA_PATH} \
       --vocab-file ${VOCAB_FILE} \
       --merge-file ${MERGE_FILE} \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend gloo \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --warmup .01 \
       --log-interval 1 \
       --exit-interval 100 \
       --save-interval 5 \
       --eval-interval 1000 \
       --use-cpu-initialization \
       --eval-iters 5 \
       --varuna --fp16 \
       # --load ${CHECKPOINT_PATH} \

set +x

