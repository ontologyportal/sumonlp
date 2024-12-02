#!/bin/bash
echo "Starting Prover ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
source ../load_configs.sh

###### FUTURE IMPROVEMENT, don't generate the file everytime, figure a way to somehow merge the file.
# Generate the SUMO.fof file or if pgs (Prover Generate Sumo.fof) flag is set
#if [[ $* == *-pgs* ]] || [ ! -f $HOME/.sigmakee/KBs/SUMO.fof ]; then
#    echo "SUMO.fof not found! Generating SUMO.fof..."
#    java  -Xmx8g -classpath   $SIGMAKEE_HOME/build/sigmakee.jar:$SIGMAKEE_HOME/lib/* com.articulate.sigma.trans.SUMOKBtoTPTPKB
#fi

# generate tptp translation of just the statement or query
bash $HOME/workspace/sumonlp/src/prover/build_tptp.sh

# add statement to the existing TPTP KB translation
cat $HOME/.sigmakee/KBs/SUMO.tptp $HOME/.sigmakee/KBs/temp-query.fof > $HOME/.sigmakee/KBs/temp-comb.fof

$HOME/workspace/vampire/vampire --input_syntax tptp -t 10 --proof tptp -qa plain --mode casc $HOME/.sigmakee/KBs/temp-comb.fof &> output_pr.txt

echo "Finished Prover ..."
