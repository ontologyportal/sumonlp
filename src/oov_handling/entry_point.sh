#!/bin/bash

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../config_paths.sh
 cp input_oov.txt output_oov.txt

echo $VOCABULARY_HOME
echo "Starting out of vocabulary handling ..."
export LD_LIBRARY_PATH="$HOME/.conda/envs/py3109_pytorch/lib"  # for running torch
python3 -u oov_handling.py      # change vocab database permissions in hamming
echo "Finished out of vocabulary proccessing ..."

