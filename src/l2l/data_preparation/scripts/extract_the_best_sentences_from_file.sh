#!/bin/bash -l

#SBATCH --job-name=extract_best_sentences
#SBATCH --output=./out/jobs/best_sentences_%j.out
#SBATCH --time=40:00:00
#SBATCH --mem=60G
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


pixi run python -u scripts/extract_the_best_sentences_from_the_file.py

