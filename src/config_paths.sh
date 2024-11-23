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
export OLLAMA_PATH="/data/angelos.toutsios.gr/Programs/ollama/bin"
export OLLAMA_LOG_PATH="/data/angelos.toutsios.gr/Programs/llamaMD/ollama_log.out"
export VAMPIRE_PATH="/data/angelos.toutsios.gr/Programs/vampire/build/"
export SIGMAKEE_HOME="/data/angelos.toutsios.gr/Programs/sigmakee"
export MODEL_HOME="/data/angelos.toutsios.gr/Programs/t5_model"
export VOCABULARY_HOME="/data/angelos.toutsios.gr/"
export CUDA_HOME="/share/apps/nvidia/cuda-12.2/lib64/"   # may not be needed
export HOST_PORT="55848"   # changes default port to unique port number, can change this
export NEW_HOME="/data/angelos.toutsios.gr/Programs"
export OLLAMA_HOST="127.0.0.1:$HOST_PORT"
# export PATH="/share/spack/gcc-10.3.0/miniconda3-23.1.0-4vp/bin:$PATH"
export PATH="/data/angelos.toutsios.gr/Programs/miniconda3/bin:$PATH"


# LOCAL ENVs
################### UPDATE THESE ####################
# export RUNNING_ON_HAMMING=false
# export CONDA_ENVIRONMENT="py312_pytorch"
# export OLLAMA_PATH="$HOME/Programs/ollama/bin"
# export OLLAMA_LOG_PATH="$HOME/Programs/llamaMD/ollama_log.out"
# export VAMPIRE_PATH="$HOME/workspace/vampire"
# export MODEL_HOME="$HOME/workspace/L2L_model/t5_model"
# export VOCABULARY_HOME="$HOME/.sumonlp"
# export OLLAMA_HOST_PORT="11434" # Used to change default port (11434) to unique port number if necessary. Default has been shown to cause problems on Hamming.
# export OLLAMA_HOST="127.0.0.1:$OLLAMA_HOST_PORT"
# export MODEL_MH="mistral" # Model to be used in metaphor handling. Common models are mistral and llama3.2
######################################################



# If PATHs don't exist on $PATH variable, then add them to the $PATH variable
[[ ":$PATH:" == *$OLLAMA_PATH* ]] || export PATH="$OLLAMA_PATH:$PATH"
[[ ":$PATH:" == *$VAMPIRE_PATH* ]] || export PATH="$VAMPIRE_PATH:$PATH"

#initialize the conda environment
# Check if the conda environment exists
eval "$(conda shell.bash hook)"
if ! conda env list | grep -q 'py3109_pytorch'; then
    # Create the environment if it doesn't exist
    conda create -n $CONDA_ENVIRONMENT python=3.9 -y
fi

# eval "$(conda shell.bash hook)"

conda activate $CONDA_ENVIRONMENT
