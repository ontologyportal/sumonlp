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
source ../config_paths.sh


# Check if Ollama server is running
if lsof -i :$HOST_PORT | grep ollama > /dev/null; then
  echo "Ollama server is running on $HOST_PORT"
else
  echo "Ollama server is not running. Starting server..."
  
  ollama serve > ./ollama_log.out 2>&1 &
  #ollama serve &
  # Give the server a chance to start up before moving to the next instruction
  sleep 2
fi