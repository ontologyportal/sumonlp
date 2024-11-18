#!/bin/bash

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../config_paths.sh
cp input_post_oov.txt output_post_oov.txt

echo "Starting out of vocabulary handling ..."
python3 -u post_oov_handling.py
echo "Finished out of vocabulary proccessing ..."

