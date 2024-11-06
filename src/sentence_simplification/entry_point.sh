#!/bin/bash
echo "Starting sentence simplification ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../config_paths.sh
../utils/start_ollama.sh

cp input_ss.txt output_ss.txt