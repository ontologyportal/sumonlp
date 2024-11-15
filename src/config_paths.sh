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



CONDA_ENVIRONMENT="py3109_pytorch"
OLLAMA_PATH="$HOME/Programs/ollama/bin"
OLLAMA_LOG_PATH="$HOME/Programs/llamaMD/ollama_log.out"
VAMPIRE_PATH="$HOME/workspace/vampire"
SIGMAKEE_HOME="$HOME/workspace/sigmakee"
MODEL_HOME="$HOME/data/workspace/L2L_model/t5_model"


# If PATHs don't exist on $PATH variable, then add them to the $PATH variable
[[ ":$PATH:" == *$OLLAMA_PATH* ]] || PATH="$OLLAMA_PATH:$PATH"
[[ ":$PATH:" == *$VAMPIRE_PATH* ]] || PATH="$VAMPIRE_PATH:$PATH"

#initialize the conda environment

# Check if the environment 'py3109_pytorch' exists
if ! conda env list | grep -q 'py3109_pytorch'; then
    # Create the environment if it doesn't exist
    conda create -n py3109_pytorch python=3.9 -y
fi

eval "$(conda shell.bash hook)"
conda activate py3109_pytorch

# Troubleshooting efforts:
export HOST_PORT="55848"

export LD_LIBRARY_PATH="$HOME/.conda/envs/py3109_pytorch/lib"  # for running torch
#export OLLAMA_HOST="127.0.0.1:11434"                           # for running ollama
export OLLAMA_HOST="127.0.0.1:$HOST_PORT"
echo $LD_LIBRARY_PATH