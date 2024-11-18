#!/bin/bash
##############################################################
# This runs tests on the entire pipeline
###############################################################

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit

cp test_set_1.txt ../policy_extracter/input_pe.txt
../main.sh
