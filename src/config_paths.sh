##############################################################
# When running this, run with
#
#   source config_path.sh
#
# This ensures it applies to the current batch session.
# Otherwise when the script is run, the PATH is only valid
# for the commands in config_path.sh
###############################################################

OLLAMA_PATH="$HOME/Programs/ollama/bin"



# If ollama/bin doesn't exist on the path, then add it to the path
[[ ":$PATH:" == *$OLLAMA_PATH* ]] || PATH="$OLLAMA_PATH:$PATH"

