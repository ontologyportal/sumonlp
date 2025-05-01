#!/bin/bash
#SBATCH --job-name=run_weirdness_detector
#SBATCH --output=/home/angelos.toutsios.gr/workspace/sumonlp/src/weirdness_detector/logs/log_%j.out  # Output log file
#SBATCH --error=/home/angelos.toutsios.gr/workspace/sumonlp/src/weirdness_detector/logs/log_%j.err   # Error log file
#SBATCH -N 1
#SBATCH --mem=60G
#SBATCH --cpus-per-task=64
#SBATCH --time=80:00:00              # Time limit (hh:mm:ss)
#SBATCH --partition=beards            # Specify the partition
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3                 # Request 3 GPU

# Load necessary environment variables
source ~/.bashrc

# Navigate to the Ollama binary directory
cd $SUMO_NLP_HOME/src/utils/


# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_NUM_PARALLEL=8
export OLLAMA_GPU_OVERHEAD=512
export OLLAMA_FLASH_ATTENTION=true

# Start the Ollama server in the background and log output
# ./ollama serve > ollama.log 2>&1 &
bash start_ollama.sh

# Give some time for the server to start
sleep 10

# Optionally check if the server is running (example for a process check)
if ! pgrep -f "ollama serve" > /dev/null; then
    echo "Ollama server failed to start. Exiting..." >&2
    exit 1
fi

# Navigate back to the weirdness detector directory
cd $SUMO_NLP_HOME/src/weirdness_detector

# Run the parallel Python script and redirect output
python3 -u ollama-model-parallel.py \
"$SUMO_NLP/src/weirdness_detector/test.txt"
