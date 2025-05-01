#!/bin/bash -l

#SBATCH --job-name=txt_to_json_dataset
#SBATCH --output=./out/jobs/Dataset_%j.out
#SBATCH --time=2:00:00
#SBATCH --mem=120G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

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


srun --time=2:00:00 python src/utils/txt_to_json.py \
/data/fsg/.sumonlp/model_generation/2025-04-25/combined-eng.txt \
/data/fsg/.sumonlp/model_generation/2025-04-25/combined-log.txt

