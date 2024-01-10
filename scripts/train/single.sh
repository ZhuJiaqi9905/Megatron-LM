#!/bin/bash

#SBATCH -p batch_block1,batch_block2,batch_block3,batch_block4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH -A llmservice_nlp_fm
#SBATCH -t 0:30:00
#SBATCH --exclusive
#SBATCH --job-name=adlr-nlp:retro-mcore
#SBATCH --dependency=singleton

# ... SBATCH -A adlr_nlp_llmnext

set -u

export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_SOCKET_IFNAME=^vlan,lo
# unset NCCL_DEBUG
export NCCL_DEBUG=INFO

REPO_DIR="/lustre/fsw/portfolios/adlr/users/lmcafee/retro/megatrons/retro-mcore-data"

######## Arguments. ########

. args_wiki.sh

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# customize / begin.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

CHECKPOINT_DIR="${RETRO_PROJECT_DIR}/checkpoints/models/c${USE_CORE}-r${ADD_RETRIEVER}-t${TP}"
TENSORBOARD_DIR="${CHECKPOINT_DIR}/tb"
LOG_DIR="${CHECKPOINT_DIR}/logs"

mkdir -p ${TENSORBOARD_DIR}
mkdir -p ${LOG_DIR}

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
# customize / end.
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

LOG_INTERVAL=5
SAVE_INTERVAL=10 # 2000 # [2000], *10000
ARGS+=" \
    --log-interval ${LOG_INTERVAL} \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --log-validation-ppl-to-tensorboard \
    --save-interval ${SAVE_INTERVAL} \
    --save ${CHECKPOINT_DIR} \
    --load ${CHECKPOINT_DIR} \
"

######## Command. ########

CMD="export PYTHONPATH=${REPO_DIR}:/home/lmcafee/src && python -u ${REPO_DIR}/pretrain_retro.py ${ARGS}"
MOUNTS="/home/lmcafee:/home/lmcafee"
MOUNTS+=",/lustre/fsw/portfolios/adlr/users/lmcafee:/lustre/fsw/portfolios/adlr/users/lmcafee"
MOUNTS+=",/lustre/fs6/portfolios/adlr/users/lmcafee:/lustre/fs6/portfolios/adlr/users/lmcafee"
IMAGE="gitlab-master.nvidia.com/adlr/megatron-lm/lmcafee/retro-process-23.04"

export CUDA_DEVICE_MAX_CONNECTIONS=1
srun -l \
     --container-image ${IMAGE} \
     --container-mounts ${MOUNTS} \
     --output=${LOG_DIR}/"%j.log" \
     sh -c "${CMD}"

# eof
