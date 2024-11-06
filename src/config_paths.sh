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

OLLAMA_PATH="$HOME/Programs/ollama/bin"2
VAMPIRE_PATH="$HOME/workspace/vampire"
SIGMAKEE_HOME="$HOME/workspace/sigmakee"
MODEL_HOME="$HOME/workspace/L2L_model/t5_model"

# If PATHs don't exist on $PATH variable, then add them to the $PATH variable
[[ ":$PATH:" == *$OLLAMA_PATH* ]] || PATH="$OLLAMA_PATH:$PATH"
[[ ":$PATH:" == *$VAMPIRE_PATH* ]] || PATH="$VAMPIRE_PATH:$PATH"

