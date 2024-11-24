#!/bin/bash
echo "Starting proof paraphrasing ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../config_paths.sh

if [ $RUNNING_ON_HAMMING == true ]; then
  module unload lib/cuda/12.2  # need to unload this because otherwise torch will be messed up. Messy fix, may need to find torch <-> ollama CUDA version incompatability
fi


# Select model to use from below, will also have to change the argument to 'OLLAMA_PATH run <>' in java code (if using java)

#ollama pull llama3.2:1b > /dev/null 2>&1
#ollama pull llama3.2:3b > /dev/null 2>&1
#ollama pull llama3.2:3b > /dev/null 2>&1
#ollama pull llama3.2:3b
#ollama pull llama3.1:8b
ollama pull $MODEL_PP > /dev/null 2>&1

echo "Running proof paraphrasing..."

java -Xmx8g -classpath   $SIGMA_SRC/build/sigmakee.jar:$SIGMA_SRC/lib/* com.articulate.sigma.trans.TPTP3ProofProcessor -s output_pr.txt &> input_pp.txt

python3 ppara.py input_pp.txt output_pp.txt "$MODEL_PP"
