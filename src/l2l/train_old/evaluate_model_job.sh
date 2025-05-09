#!/bin/bash
#SBATCH --job-name=evaluate_t5_model
#SBATCH --output=./logs/log_%j.out  # Output log file
#SBATCH --error=./logs/log_%j.err   # Error log file
#SBATCH -N 1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=32
#SBATCH --time=80:00:00              # Time limit (hh:mm:ss)
#SBATCH --partition=genai            # Specify the partition
#SBATCH --gres=gpu:1                 # Request 1 GPU

python -u evaluate_model.py
