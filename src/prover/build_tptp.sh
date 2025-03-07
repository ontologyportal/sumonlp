#!/bin/bash
java -Xmx8g -classpath $SIGMA_SRC/build/sigmakee.jar:$SIGMA_SRC/lib/* com.articulate.sigma.trans.SUMOformulaToTPTPformula -g "$(< $ONTOLOGYPORTAL_GIT/sumonlp/src/prover/input_pr.txt)" > $SIGMA_HOME/KBs/temp.fof
tail -n 1 $SIGMA_HOME/KBs/temp.fof > $SIGMA_HOME/KBs/temp-query.fof
sed -i '1s/^/fof(query,conjecture,/' $SIGMA_HOME/KBs/temp-query.fof
printf ")." >> $SIGMA_HOME/KBs/temp-query.fof
rm $SIGMA_HOME/KBs/temp.fof 
rm $SIGMA_HOME/KBs/temp-comb.fof
