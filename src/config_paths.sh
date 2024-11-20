#!/bin/bash
##############################################################
# When running this, run with
#
#   source config_path.sh
#
# This ensures it applies to the current batch session.
# Otherwise when the script is run, the PATH is only valid
# for the commands in config_path.sh
###############################################################

export CONDA_ENVIRONMENT="py3109_pytorch"
export OLLAMA_PATH="$HOME/Programs/ollama/bin"
export OLLAMA_LOG_PATH="$HOME/Programs/llamaMD/ollama_log.out"
export VAMPIRE_PATH="$HOME/workspace/vampire"
export SIGMAKEE_HOME="$HOME/workspace/sigmakee"
export MODEL_HOME="$HOME/data/workspace/L2L_model/t5_model"
export VOCABULARY_HOME="/data/angelos.toutsios.gr/"
export CUDA_HOME="/share/apps/nvidia/cuda-12.2/lib64/"   # may not be needed
export HOST_PORT="55848"   # changes default port to unique port number, can change this
export OLLAMA_HOST="127.0.0.1:$HOST_PORT"


# If PATHs don't exist on $PATH variable, then add them to the $PATH variable
[[ ":$PATH:" == *$OLLAMA_PATH* ]] || PATH="$OLLAMA_PATH:$PATH"
[[ ":$PATH:" == *$VAMPIRE_PATH* ]] || PATH="$VAMPIRE_PATH:$PATH"

#initialize the conda environment
# Check if the conda environment exists
if ! conda env list | grep -q 'py3109_pytorch'; then
    # Create the environment if it doesn't exist
    conda create -n $CONDA_ENVIRONMENT python=3.9 -y
fi

eval "$(conda shell.bash hook)"

conda activate $CONDA_ENVIRONMENT
