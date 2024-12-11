#!/bin/bash
####################################################################################################
# Checks to see if an o-llama server is already running.
# If one is not, then it will start an instance.
#
# EXPLANATION:
# lsof -i :11434:
# This command lists all open files (including network connections) that
# are using port 11434, which is the default port used by Ollama.
#
# grep ollama:
# This command filters the output of lsof to only show lines that contain the word "ollama".
#
# > /dev/null:
# This redirects the output of the grep command to /dev/null, which effectively discards it.
# This is done because we only care about the exit status of the command.
####################################################################################################

# Gets the path to this script, so this script can be run from any location.
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
# source ../config_paths.sh

# Define required model names
MODEL_1="llama3.1:8b"
MODEL_2="llama3.2:latest"

# Function to check and pull model if necessary
check_and_pull_model() {
    local model_name=$1
    if ! ollama list | grep -q "$model_name"; then
        echo "Model '$model_name' not found. Pulling it now..."
        ollama pull "$model_name"
    else
        echo "Model '$model_name' is already installed."
    fi
}

# Check if Ollama server is running
if lsof -i :$OLLAMA_HOST_PORT | grep ollama > /dev/null; then
  echo "Ollama server is running on $OLLAMA_HOST"
else
  echo "Ollama server is not running. Starting server..."
  if [[ $SUMO_NLP_RUNNING_ON_HAMMING == true ]]; then
    # module purge
    # module load lib/cuda/12.2  # needs this module for running server with GPU
  fi
  ollama serve > ../ollama_log.out 2>&1 &
  # Give the server a chance to start up before moving to the next instruction
  sleep 1
fi

# Check and pull both models
check_and_pull_model "$MODEL_1"
check_and_pull_model "$MODEL_2"