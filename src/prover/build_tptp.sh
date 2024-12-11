#!/bin/bash
java -Xmx8g -classpath $SIGMA_SRC/build/sigmakee.jar:$SIGMA_SRC/lib/* com.articulate.sigma.trans.SUMOformulaToTPTPformula -g "$(< $SUMO_NLP_HOME/src/prover/input_pr.txt)" > $HOME/.sigmakee/KBs/temp.fof
tail -n 1 $HOME/.sigmakee/KBs/temp.fof > $HOME/.sigmakee/KBs/temp-query.fof
sed -i '1s/^/fof(query,conjecture,/' $HOME/.sigmakee/KBs/temp-query.fof
printf ")." >> $HOME/.sigmakee/KBs/temp-query.fof
rm $HOME/.sigmakee/KBs/temp.fof
rm $HOME/.sigmakee/KBs/temp-comb.fof
