#!/bin/bash
#SBATCH --job-name=create_voc_from_coca
#SBATCH --output=/home/angelos.toutsios.gr/workspace/sumonlp/src/oov_handling/logs/log_%j.out  # Output log file
#SBATCH --error=/home/angelos.toutsios.gr/workspace/sumonlp/src/oov_handling/logs/log_%j.err   # Error log file
#SBATCH -N 1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=64
#SBATCH --time=80:00:00             # Time limit (hh:mm:ss)


python3 -u vocabulary_from_coca.py # -u is for printing the outcome directly and not buffer it