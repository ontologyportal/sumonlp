#!/bin/bash
#SBATCH --job-name=custom_t5_training
#SBATCH --output=/home/angelos.toutsios.gr/data/Thesis_dev/nmt_model_train_custom/logs/log_%j.out  # Output log file
#SBATCH --error=/home/angelos.toutsios.gr/data/Thesis_dev/nmt_model_train_custom/logs/log_%j.err   # Error log file
#SBATCH -N 1
#SBATCH --mem=250G
#SBATCH --cpus-per-task=32
#SBATCH --time=80:00:00              # Time limit (hh:mm:ss)
#SBATCH --partition=genai            # Specify the partition
#SBATCH --gres=gpu:1                 # Request 1 GPU

# Start logging GPU usage
watch -n 60 nvidia-smi > gpu_usage.log &  # Log every 5 seconds

python -u train.py
