#!/bin/bash
echo "Starting sentence simplification ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../load_configs.sh
../utils/start_ollama.sh > logs/ollama.log 2>&1 &

ollama pull llama3.1:8b-instruct-q8_0
ollama create simplify_model -f ./models/Modelfile_llama3.1_8b-instruct-q8_0

cp input_ss.txt output_ss.txt

# python3 main.py
echo $MODEL_SS
python3 main.py input_ss.txt output_ss.txt "$MODEL_SS"


echo "Finished sentence simplification ..."