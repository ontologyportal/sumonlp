#!/bin/bash -l

#SBATCH --job-name=L2L
#SBATCH --output=./out/jobs/L2L_%j.out
#SBATCH --time=90:00:00
#SBATCH --mem=240G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --partition=genai

# NOTES:
# SBATCH --ntasks-per-node and --gres, training.devices need to be the same
# SBATCH --cpus-per-task and datamodule.num_workers should be the same
# It appears that specifying 'srun' before 'python' is necessary
# You need to re-specify --time to srun, or else your job will be killed after a short amount of time
# If you want to run in debug mode, run single GPU

OUTPUT_DIR=out/$(date +%Y-%m-%d_%H-%M-%S)
mkdir -p ${OUTPUT_DIR}

## Pixi ##
eval "$(pixi shell-hook -s bash)"

## Debugging ##
export HYDRA_FULL_ERROR=1

export TOKENIZERS_PARALLELISM=false

## This helps reduce memory fragmentation.
# PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

## Multi-node training ##

# InfiniBand
# export NCCL_IB_HCA=ib0

# If InfiniBand is having issues
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IB_DISABLE=1

# Log GPU memory usage every 30 seconds
( while true; do nvidia-smi >> ${OUTPUT_DIR}/gpu_memory.log; sleep 600; done ) &

# Log SLURM memory usage every 30 seconds
( while true; do sstat -j $SLURM_JOB_ID --format=JobID,AveRSS%20,MaxRSS%20,MaxVMSize%20 >> ${OUTPUT_DIR}/ram_memory.log; sleep 600; done ) &

#srun --time=2:00:00 python -m debugpy --wait-for-client --listen 0.0.0.0:54327 src/train.py \
srun python -u src/train.py \
--config-dir configs \
--config-name config.yaml \
'hydra.run.dir=${paths.output_dir}' \
'datamodule.batch_size=32' \
'datamodule.num_workers=8' \
'module.lr=1e-4' \
'module.patience=5' \
'module.weight_decay=0.001' \
'module.dropout=0.2' \
'module.use_pretrained=true' \
'module.warm_up_step=0' \
'module.sumo_term_penalty_weight=0' \
'trainer.num_nodes=1' \
'trainer.precision=bf16-mixed' \
'trainer.max_epochs=5' \
'trainer.accelerator=auto' \
'trainer.accumulate_grad_batches=8' \
'trainer.strategy=auto' \
'trainer.devices=auto' \
'trainer.sync_batchnorm=true' \
'trainer.gradient_clip_val=0.5' \
'trainer.gradient_clip_algorithm=norm' \
'trainer.profiler=simple' \
'callback_early_stopping.patience=5' \
'paths.output_dir='${OUTPUT_DIR} \
"paths.data_dir=${SUMO_NLP_HOME}/src/l2l/train_new/data/" \
'paths.input_file=/home/angelos.toutsios.gr/workspace/sumonlp/src/l2l/train_new/data/test10k.json' \
'paths.data_name=test10k' \

