#!/bin/bash
# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit

./utils/start_ollama.sh

FLOW_FILE="flow.txt"
> "$FLOW_FILE"
echo -e "\nInput" >> "$FLOW_FILE"
cat policy_extracter/input_pe.txt >> "$FLOW_FILE"


bash policy_extracter/entry_point.sh
cp policy_extracter/output_pe.txt metaphor_handling/input_mh.txt
echo -e "\n\nAfter policy extractor" >> "$FLOW_FILE"
cat policy_extracter/output_pe.txt >> "$FLOW_FILE"

bash metaphor_handling/entry_point.sh
cp metaphor_handling/output_mh.txt sentence_simplification/input_ss.txt
echo -e "\n\nAfter metaphor handling" >> "$FLOW_FILE"
cat metaphor_handling/output_mh.txt >> "$FLOW_FILE"

bash sentence_simplification/entry_point.sh
cp sentence_simplification/output_ss.txt oov_handling/input_oov.txt
echo -e "\nAfter sentence simplification" >> "$FLOW_FILE"
cat sentence_simplification/output_ss.txt >> "$FLOW_FILE"

bash oov_handling/entry_point.sh
cp oov_handling/output_oov.txt l2l/input_l2l.txt
echo -e "\nAfter OOV handling" >> "$FLOW_FILE"
cat oov_handling/output_oov.txt >> "$FLOW_FILE"

bash l2l/entry_point.sh
cp l2l/output_l2l.txt oov_handling/input_post_oov.txt
echo -e "\nAfter L2L" >> "$FLOW_FILE"
cat l2l/output_l2l.txt >> "$FLOW_FILE"

bash oov_handling/entry_point_postprocessing.sh
cp oov_handling/output_post_oov.txt prover/input_pr.txt
echo -e "\nAfter OOV post-processing" >> "$FLOW_FILE"
cat oov_handling/output_post_oov.txt >> "$FLOW_FILE"

bash prover/entry_point.sh
cp prover/output_pr.txt ppara/input_pp.txt
echo -e "\nAfter prover" >> "$FLOW_FILE"
cat prover/output_pr.txt >> "$FLOW_FILE"

bash ppara/entry_point.sh
echo -e "\nAfter proof paraphrasing" >> "$FLOW_FILE"
cat ppara/output_pp.txt >> "$FLOW_FILE"

