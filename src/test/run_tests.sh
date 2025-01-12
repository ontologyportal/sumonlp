#!/bin/bash
##############################################################
# This runs tests on the entire pipeline
###############################################################

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit

# uncomment to run on all the test sentences
# leave commented out to run only on the current contents of input_pe.txt
#cat test*.txt >> ../policy_extracter/input_pe.txt

../run_pipeline.sh

sed -n '/After OOV post-processing/,$p' ../flow.txt

