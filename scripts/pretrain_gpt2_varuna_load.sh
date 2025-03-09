#! /bin/bash

model=${1:-"gpt3_350M"}
nstages=${2:-1}
mbs=${3:-8}
total_gpus=${4:-16}
MASTER_ADDR=${5:-"172.31.44.99"}

gbs=2048
gpus_per_node=1


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
elif [[ "${model}" == "gpt3_13B" ]]; then
       NUM_LAYERS=40
       HIDDEN_SIZE=5140
       NUM_ATTENTION_HEADS=40
else
       echo "Don't have model ${model}"
       exit -1
fi

DATA_PATH=/mnt/varuna/dataset/gpt-dataset-simplewiki/meg-gpt2_text_document
VOCAB_FILE=/mnt/varuna/dataset/gpt2-vocab.json
MERGE_FILE=/mnt/varuna/dataset/gpt2-merges.txt
CHECKPOINT_PATH=/mnt/varuna/checkpoints/${model}/${total_gpus}

# rm _tmp_*
# rm -rf ${CHECKPOINT_PATH}/*

# --nstages ${nstages} --chunk_size ${mbs} \

export GLOO_SOCKET_IFNAME=enp216s0np0 && \
python3 -m varuna.run_varuna \
       --manager_ip $MASTER_ADDR \
       --nstages ${nstages} --chunk_size ${mbs} \
       --batch_size ${gbs} \
       --total_gpus ${total_gpus} \
       --gpus_per_node ${gpus_per_node} \
       --log_dir res/ssh_log_${total_gpus}_${model}_${nstages}_${mbs}_load \
       --no_morphing pretrain_gpt2.py \
       --num-layers $NUM_LAYERS \
       --hidden-size $HIDDEN_SIZE \
       --num-attention-heads $NUM_ATTENTION_HEADS \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 1 \
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
       --save-interval 3 \
       --eval-interval 1000 \
       --use-cpu-initialization \
       --eval-iters 5 \
       --varuna --fp16 \
       --fp16-lm-cross-entropy \
       --load ${CHECKPOINT_PATH}

set +x

