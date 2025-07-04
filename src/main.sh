#!/bin/bash
# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit

source load_configs.sh
BLACK=`tput setaf 0`
RED=`tput setaf 1`
GREEN=`tput setaf 2`
YELLOW=`tput setaf 3`
BLUE=`tput setaf 4`
MAGENTA=`tput setaf 5`
CYAN=`tput setaf 6`
WHITE=`tput setaf 7`
RESET=`tput sgr0`


echo "Starting SUMO Language to Logic conversion. type '${YELLOW}help${RESET}' for command list ..."
while true; do
    # Read user input
    read -p "${MAGENTA}>>>${RESET} " input
    if [ -n "$input" ]; then # only add it if it is not empty.
      echo $input >> command_history.txt
    fi
    # Extract the first word
    command=$(echo $input | awk '{print $1}')

    # Execute commands based on the first word
    case $command in
        ask)
            ask_value=${input:4} # gets the substring from character position 4 to the end.
            echo "Asking $ask_value"
            echo $ask_value > policy_extracter/input_pe.txt
            bash run_pipeline.sh
            bash utils/HOLtoTFF.sh prover/input_pr.txt
            question_KIF=$(cat prover/input_pr.txt)
            echo $question_KIF
            bash utils/add_SUMO_NLP_to_config_dot_xml_if_needed.sh
            question_TPTP=$(java -Xmx8g -classpath $ONTOLOGYPORTAL_GIT/sigmakee/build/sigmakee.jar:$ONTOLOGYPORTAL_GIT/sigmakee/lib/* com.articulate.sigma.trans.SUMOformulaToTPTPformula -g "$question_KIF" | tail -n 1)
            echo $question_TPTP
            cat $SIGMA_HOME/KBs/SUMO.tptp > $SIGMA_HOME/KBs/SUMO_NLP_QUERY.tptp
            echo "fof(name1,conjecture, $question_TPTP)." >> $SIGMA_HOME/KBs/SUMO_NLP_QUERY.tptp
            vampire --input_syntax tptp -t 60 --proof tptp -qa plain --mode casc $SIGMA_HOME/KBs/SUMO_NLP_QUERY.tptp | tee prover/output_pr.txt
            ;;
        add)
            add_value=${input:4} # gets the substring from character position 4 to the end.
            txt_file=false;
            echo "Adding $add_value to the knowledge base."
            if [[ "${add_value: -4}" == ".txt" ]]; then
                txt_file=true;
                echo "Text file: $add_value"
                cat $add_value > policy_extracter/input_pe.txt
            else
                echo $add_value > policy_extracter/input_pe.txt
            fi
            bash run_pipeline.sh
            bash utils/HOLtoTFF.sh prover/input_pr.txt
            cat prover/input_pr.txt >> $SIGMA_HOME/KBs/SUMO_NLP.kif
            bash utils/add_SUMO_NLP_to_config_dot_xml_if_needed.sh
            if [ -e "$SIGMA_HOME/KBs/SUMO.tptp" ] && [ $txt_file == false ]; then
                axiom_KIF=$(cat prover/input_pr.txt)
                echo $axiom_KIF
                axiom_TPTP=$(java -Xmx8g -classpath $ONTOLOGYPORTAL_GIT/sigmakee/build/sigmakee.jar:$ONTOLOGYPORTAL_GIT/sigmakee/lib/* com.articulate.sigma.trans.SUMOformulaToTPTPformula -g "$axiom_KIF" | tail -n 1)
                echo $axiom_TPTP
                echo "fof(name1,axiom, $axiom_TPTP)." >> $SIGMA_HOME/KBs/SUMO.tptp
            else
                java -Xmx8g -classpath $SIGMA_SRC/build/sigmakee.jar:$SIGMA_SRC/lib/* com.articulate.sigma.trans.SUMOKBtoTPTPKB
            fi
            #bash prover/entry_point.sh
            echo "Added $add_value to the knowledge base."
            ;;
        add_lbl)
            bash utils/add_lbl.sh ${input:8}
            ;;
        runstandardtestset|rsts)
            bash utils/add_lbl.sh test/standard.txt
            bash test/check_SUOKIF_syntax.sh test/add_lbl_output.txt -c
            python3 test/check_SUOKIF_types.py test/SUOKIF_Syntax_Check.csv test/SUOKIF_Type_Check_standard_test_set.csv
            ;;
        runrandtestset|rrts)
            output_file="test/random_test_set.txt"
            lines_to_extract=125
            shuf -n $lines_to_extract "test/EntireEconomicCorpus.txt" > "$output_file"
            echo "$lines_to_extract random lines from test/EntireEconomicCorpus.txt have been saved to $output_file"
            cat $output_file
            bash utils/add_lbl.sh $output_file
            bash test/check_SUOKIF_syntax.sh test/add_lbl_output.txt -c
            python3 test/check_SUOKIF_types.py test/SUOKIF_Syntax_Check.csv test/SUOKIF_Type_Check_random_test_set.csv
            ;;
        generateWordNetKif|gwnk)
            python3 utils/genWN_KIF.py
            cp $ONTOLOGYPORTAL_GIT/sumo/WN_Subsuming_Mappings.kif $SIGMA_HOME/KBs/WN_Subsuming_Mappings.kif
            echo "Results also copied to $SIGMA_HOME/KBs/WN_Subsuming_Mappings.kif"
            ;;
        clear)
            rm -f $SIGMA_HOME/KBs/SUMO_NLP.kif
            rm -f $SIGMA_HOME/KBs/SUMO.fof
            rm -f $SIGMA_HOME/KBs/SUMO.tptp
            rm -f $SIGMA_HOME/KBs/SUMO_NLP_QUERY.fof
            rm -f $SIGMA_HOME/KBs/*.ser
            echo "Knowledge base has been cleared."
            ;;
        list|kb)
            if [ -e "$SIGMA_HOME/KBs/SUMO_NLP.kif" ]; then
              echo "Knowledge Base: "
              cat $SIGMA_HOME/KBs/SUMO_NLP.kif
            fi
            echo -e "\n"
            ;;
        prover|test)
            if [[ $input == test* ]]; then
                time_value=${input:6}
            else
                time_value=${input:8}
            fi
            time_value=${input:6}
            bash utils/add_SUMO_NLP_to_config_dot_xml_if_needed.sh
            java -Xmx8g -classpath $SIGMA_SRC/build/sigmakee.jar:$SIGMA_SRC/lib/* com.articulate.sigma.trans.SUMOKBtoTPTPKB
            vampire --input_syntax tptp $time_value --proof tptp -qa plain --mode casc $SIGMA_HOME/KBs/SUMO.tptp | tee prover/output_pr.txt
            ;;
        proofvisual|pv)
            file="prover/output_pr.txt"
            if [ -f "$file" ]; then
                java -Xmx8g -classpath $ONTOLOGYPORTAL_GIT/sigmakee/build/sigmakee.jar:$ONTOLOGYPORTAL_GIT/sigmakee/lib/* com.articulate.sigma.trans.TPTP3ProofProcessor -f prover/output_pr.txt
                dot $CATALINA_HOME/webapps/sigma/graph/proof.dot -Tgif > prover/proof.gif
                echo "Visualization has been saved to prover/proof.gif"
                xdg-open prover/proof.gif
            else
                echo "Their is no prover output to visualize. Proof should be saved to prover/prover_output.txt"
            fi
            ;;
        last|flow)
            cat flow.txt
            ;;
        "")
            ;;
        mh)
            sentence_value=${input:3}
            echo $sentence_value > metaphor_handling/input_mh.txt
            bash metaphor_handling/entry_point.sh > metaphor_handling/logs/mh_log.log
            cat metaphor_handling/output_mh.txt
            ;;
        ss)
            sentence_value=${input:3}
            echo $sentence_value > sentence_simplification/input_ss.txt
            bash sentence_simplification/entry_point.sh > sentence_simplification/logs/ss_log.log
            echo -e "\nSentence simplified:"
            cat sentence_simplification/output_ss.txt
            ;;
        l2l)
            sentence_value=${input:3}
            echo $sentence_value > l2l/input_l2l.txt
            bash l2l/entry_point.sh
            cat l2l/output_l2l.txt
            ;;
        oov)
            sentence_value=${input:4}
            echo $sentence_value > oov_handling/input_oov.txt
            bash oov_handling/entry_point.sh > oov_handling/logs/oov_handling.log
            echo "Out of vocabulary output: "
            cat oov_handling/output_oov.txt
            cat oov_handling/output_oov.txt > l2l/input_l2l.txt
            bash l2l/entry_point.sh > l2l/l2l_log.log
            echo "L2l output: "
            cat l2l/output_l2l.txt
            cat l2l/output_l2l.txt > oov_handling/input_post_oov.txt
            bash oov_handling/entry_point_postprocessing.sh
            cat oov_handling/output_post_oov.txt
            ;;
        history|hist)
            cat command_history.txt
            ;;
        help|man)
            printf "\n\n=================================================================\n"
            printf "\nValid commands are ask, add, clear or quit\n"
            printf '\n"ask" a question from the knowledge base.\n  Example: "ask Does CO2 cause global warming?"\n'
            printf '\n"add" will append a new sentence or file to the knowledge base.\n  Example: "add CO2 causes global warming."\n'
            printf '  Example: "add climate_facts.txt"\n'
            printf '\n"add_lbl adds a text file line by line, rather than the entire file all at once. Output is saved to add_lbl_output.txt. The add_lbl_output.txt is tested for valid syntax and results saved to SUOKIF_Syntax_Check.csv \n"'
            printf '  Example: "add_lbl test/standard.txt"\n'
            printf '\n"rsts": run standard test set. This will take test/standard.txt, add_lbl, and then syntax and type check.\n'
            printf '\n"rrts": run random test set will generate a random test set of 100 sentences, drawn from test/EntireEconomicCorpus.txt, add lbl, then syntax and type check.\n'
            printf '\n"gwnk" Generates WordNet .kif file, complete with subclass, documentation, and termFormat statements. Output is saved to workspace/sumo, and .sigmakee/KBs/\n'
            printf '\n"clear" will completely clear the knowledge base.\n  Example: "clear"\n'
            printf '\n"last" will show the progression through the pipeline of the last added sentence or file.\n  Example: "last"\n'
            printf '\n"list" or "kb" will display the knowledge base.\n  Example: "list"\n'
            printf '\n"prover" or "test" will run the Vampire prover on the knowledge base, searching for contradictions. Default is 60 seconds.\n  Example: "test -t 40" # runs for 40 seconds\n  Example: "test" # runs for 60 seconds\n'
            printf '\n"proofvisual" or "pv" will create a visualization of the most recent proof\n'
            printf '\n"mh" will run just the metaphor translation portion of the pipeline.\n  Example: "mh The car flew past the barn."\n'
            printf '\n"ss" will run just the sentence simplification portion of the pipeline.\n  Example: "ss He who knows not, knows not he knows not, is a fool, shun him."\n'
            printf '\n"oov" will run just the out of vocabulary handling portion of the pipeline.\n  Example: "oov Bartholemew used the doohicky as a dinglehopper."\n'
            printf '\n"l2l" will run just the language to logic portion of the pipeline.\n  Example: "l2l A bird is a subclass of animal."\n'
            printf '\n"hist" or "history" will print the commands run.\n  Example: "hist"\n'
            printf '\n"quit" will exit the interface.\n  Example: "quit"\n'
            printf "\n=================================================================\n\n"
            ;;
        quit|exit)
            echo "Good bye."
            break
            ;;
        *)
            echo "Invalid command. Please use 'ask', 'add', 'add_lbl', 'grts', 'rrts', 'gwnp', 'proofvisual', 'clear', 'list', 'test', 'mh', 'ss', 'oov', 'l2l', 'help', 'hist' or 'quit'."
            ;;
    esac
done
