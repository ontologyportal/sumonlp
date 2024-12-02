#!/bin/bash
echo "Starting metaphor handling ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../load_configs.sh

if [[ "$SUMO_NLP_RUNNING_ON_HAMMING" == true ]]; then
  module unload lib/cuda/12.2  # need to unload this because otherwise torch will be messed up. Messy fix, may need to find torch <-> ollama CUDA version incompatability
fi
echo "Running Metaphor detector..."
python3 metaphor_detect_pipeline.py


# Select model to use from below, will also have to change the argument to 'OLLAMA_PATH run <>' in java code (if using java)

#ollama pull llama3.2:1b > /dev/null 2>&1
#ollama pull llama3.2:3b > /dev/null 2>&1
#ollama pull llama3.2:3b > /dev/null 2>&1
#ollama pull llama3.2:3b
#ollama pull llama3.1:8b
#ollama pull $MODEL_MH > /dev/null 2>&1

echo "Running Metaphor translator..."


# Java and python metaphor translators perform the same task: parse and interface with ollama

# For use in compilation of java code
#module load lang/java/8.141-oracle > /dev/null 2>&1
#javac LlamaMTrans.java
#java LlamaMTrans output_md.txt output_mh.txt

python3 metaphor_trans.py output_md.txt output_mh.txt "$MODEL_MH"