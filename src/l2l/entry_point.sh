#!/bin/bash
echo "Starting Language to Logic translation ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../config_paths.sh

export LD_LIBRARY_PATH="$HOME/.conda/envs/py3109_pytorch/lib"  # for running torch
python reference/inference.py $MODEL_HOME

echo "Finished Language to Logic translation ..."