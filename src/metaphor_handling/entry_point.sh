#!/bin/bash
echo "Starting metaphor handling ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../config_paths.sh


#SBATCH --job-name=sumo_nlp_training_smallwork_1000
#SBATCH --output=/home/roberto.milanese/scripts/log_%j.out  # Output log file
#SBATCH --error=/home/roberto.milanese/scripts/log_%j.err   # Error log file
#SBATCH -N 1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00             # Time limit (hh:mm:ss)
#SBATCH --partition=genai
#SBATCH --gres=gpu:1               # Request 1 GPU


echo "Running Metaphor detector..."
export LD_LIBRARY_PATH="$HOME/.conda/envs/py3109_pytorch/lib"  # for running torch
python3 metaphor_detect_pipeline.py



export OLLAMA_HOST="127.0.0.1:55848"
#ollama serve > ./ollama_log.out 2>&1 &
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/share/apps/nvidia/cuda-12.2/lib64   # for running ollama
../utils/start_ollama.sh
echo "Ollama Host: $OLLAMA_HOST"

# Select model to use from below, will also have to change the argument to 'OLLAMA_PATH run <>' in java code 
#ollama pull llama3.2:1b > /dev/null 2>&1
#ollama pull llama3.2:3b > /dev/null 2>&1
#ollama pull llama3.2:3b > /dev/null 2>&1
#ollama pull llama3.2:3b
#ollama pull llama3.1:8b
#ollama pull mistral > /dev/null 2>&1
ollama pull mistral


echo "Running Metaphor translator..."
echo "Ollama Host: $OLLAMA_HOST"

# Java and python metaphor translators perform the same task: parse and interface with ollama

# For use in compilation of java code
#module load lang/java/8.141-oracle > /dev/null 2>&1
#javac LlamaMTrans.java
#java LlamaMTrans output_md.txt output_mh.txt


python3 metaphor_trans.py output_md.txt output_mh.txt mistral