#!/bin/bash
echo "Starting SUMO Language to Logic conversion ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit


source config_paths.sh

./utils/start_ollama.sh

bash policy_extracter/entry_point.sh
cp policy_extracter/output_pe.txt metaphor_handling/input_mh.txt

bash metaphor_handling/entry_point.sh
cp metaphor_handling/output_mh.txt sentence_simplification/input_ss.txt

bash sentence_simplification/entry_point.sh
cp sentence_simplification/output_ss.txt oov_handling/input_oov.txt

bash oov_handling/entry_point.sh
cp oov_handling/output_oov.txt l2l/input_l2l.txt

bash l2l/entry_point.sh
cp l2l/output_l2l.txt oov_handling/input_post_oov.txt

bash oov_handling/entry_point_postprocessing.sh
cp oov_handling/output_post_oov.txt prover/input_pr.txt

bash prover/entry_point.sh

echo "Done!"
