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


################### UPDATE THESE ####################
export RUNNING_ON_HAMMING=false
export CONDA_ENVIRONMENT="py312_pytorch"
export OLLAMA_PATH="$HOME/Programs/ollama/bin"
export OLLAMA_LOG_PATH="$HOME/Programs/llamaMD/ollama_log.out"
export VAMPIRE_PATH="$HOME/workspace/vampire"
export MODEL_HOME="$HOME/workspace/L2L_model/t5_model"
export VOCABULARY_HOME="$HOME/.sumonlp"
export OLLAMA_HOST_PORT="11434" # Used to change default port (11434) to unique port number if necessary. Default has been shown to cause problems on Hamming.
export OLLAMA_HOST="127.0.0.1:$OLLAMA_HOST_PORT"
export MODEL_MH="mistral" # Model to be used in metaphor handling. Common models are mistral and llama3.2
export MODEL_PP="llama3.2" # Model to be used in metaphor handling. Common models are mistral and llama3.2
export MODEL_SS="llama3.2" # Model to be used in sentence simplification. Common models are mistral and llama3.2
######################################################




################## DON'T TOUCH THESE ######################
# If PATHs don't exist on $PATH variable, then add them to the $PATH variable
[[ ":$PATH:" == *$OLLAMA_PATH* ]] || export PATH="$OLLAMA_PATH:$PATH"
[[ ":$PATH:" == *$VAMPIRE_PATH* ]] || export PATH="$VAMPIRE_PATH:$PATH"

#initialize the conda environment
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENVIRONMENT
