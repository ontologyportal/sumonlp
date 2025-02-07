#!/bin/bash
#SBATCH --job-name=voc_from_sentences
#SBATCH --output=/home/angelos.toutsios.gr/data/Train_T5_Model/data/scripts/logs/log_%j.out  # Output log file
#SBATCH --error=/home/angelos.toutsios.gr/data/Train_T5_Model/data/scripts/logs/log_%j.err   # Error log file
#SBATCH -N 1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=64
#SBATCH --time=48:00:00             # Time limit (hh:mm:ss)


python3 -u create_voc_from_sentences.py # -u is for printing the outcome directly and not buffer it