#!/bin/bash
echo "Starting sentence simplification ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../load_configs.sh
../utils/start_ollama.sh > logs/ollama.log 2>&1 &

cp input_ss.txt output_ss.txt

# python3 main.py
python3 main.py input_ss.txt output_ss.txt "$MODEL_SS"


echo "Finished sentence simplification ..."