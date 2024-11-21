#!/bin/bash
echo "Starting Prover ..."

# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
# source ../config_paths.sh

###### FUTURE IMPROVEMENT, don't generate the file everytime, figure a way to somehow merge the file.
# Generate the SUMO.fof file or if pgs (Prover Generate Sumo.fof) flag is set
#if [[ $* == *-pgs* ]] || [ ! -f $HOME/.sigmakee/KBs/SUMO.fof ]; then
#    echo "SUMO.fof not found! Generating SUMO.fof..."
#    java  -Xmx8g -classpath   $SIGMAKEE_HOME/build/sigmakee.jar:$SIGMAKEE_HOME/lib/* com.articulate.sigma.trans.SUMOKBtoTPTPKB
#fi

cp input_pr.txt /data/angelos.toutsios.gr/Programs/.sigmakee/KBs/SUMO_NLP.kif

java  -Xmx8g -classpath   $SIGMAKEE_HOME/build/sigmakee.jar:$SIGMAKEE_HOME/lib/* com.articulate.sigma.trans.SUMOKBtoTPTPKB
vampire --input_syntax tptp -t 10 --proof tptp -qa plain --mode casc /data/angelos.toutsios.gr/Programs/.sigmakee/KBs/SUMO.fof

cp /data/angelos.toutsios.gr/Programs/.sigmakee/KBs/SUMO.tptp output_pr.txt

echo "Finished Prover ..."