#!/bin/bash
# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path"
# Strip \r from the end of every line.
sed -i 's/\r//g' *.sh
sed -i 's/\r//g' ../*.sh
sed -i 's/\r//g' ../metaphor_handling/*.sh
sed -i 's/\r//g' ../oov_handling/*.sh
sed -i 's/\r//g' ../policy_extracter/*.sh
sed -i 's/\r//g' ../prover/*.sh
sed -i 's/\r//g' ../sentence_simplification/*.sh
sed -i 's/\r//g' ../test/*.sh
sed -i 's/\r//g' ../l2l/*.sh