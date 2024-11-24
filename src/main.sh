#!/bin/bash
# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit

source config_paths.sh
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
    echo $input >> command_history.txt
    # Extract the first word
    command=$(echo $input | awk '{print $1}')

    # Execute commands based on the first word
    case $command in
        ask)
            ask_value=${input:4} # gets the substring from character position 4 to the end.
            echo "Asking $ask_value"
            echo $ask_value > policy_extracter/input_pe.txt
            bash run_pipeline.sh
            cat prover/input_pr.txt >> $HOME/.sigmakee/KBs/SUMO_NLP.kif
            question_KIF=$(cat prover/input_pr.txt)
            echo $question_KIF
            question_TPTP=$(java -Xmx8g -classpath $ONTOLOGYPORTAL_GIT/sigmakee/build/sigmakee.jar:$ONTOLOGYPORTAL_GIT/sigmakee/lib/* com.articulate.sigma.trans.SUMOformulaToTPTPformula -g "$question_KIF" | tail -n 1)
            cat $HOME/.sigmakee/KBs/SUMO.fof > $HOME/.sigmakee/KBs/SUMO_NLP_QUERY.fof
            echo "fof(name1,conjecture, $question_TPTP)." >> $HOME/.sigmakee/KBs/SUMO_NLP_QUERY.fof
            vampire --input_syntax tptp -t 10 --proof tptp -qa plain --mode casc $HOME/.sigmakee/KBs/SUMO_NLP_QUERY.fof > prover/output_pr.txt
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
            cat prover/input_pr.txt >> $HOME/.sigmakee/KBs/SUMO_NLP_KB.kif
            cat $HOME/.sigmakee/KBs/SUMO_NLP_KB.kif > $HOME/.sigmakee/KBs/SUMO_NLP.kif
            if [ -e "$HOME/.sigmakee/KBs/SUMO.fof" ] && [ $txt_file == false ]; then
                axiom_KIF=$(cat prover/input_pr.txt)
                echo $axiom_KIF
                axiom_TPTP=$(java -Xmx8g -classpath $ONTOLOGYPORTAL_GIT/sigmakee/build/sigmakee.jar:$ONTOLOGYPORTAL_GIT/sigmakee/lib/* com.articulate.sigma.trans.SUMOformulaToTPTPformula -g "$axiom_KIF" | tail -n 1)
                echo $axiom_TPTP
                echo "fof(name1,axiom, $axiom_TPTP)." >> $HOME/.sigmakee/KBs/SUMO.fof
            else
                bash prover/build_tptp.sh # Builds the whole thing.
            fi
            #bash prover/entry_point.sh
            echo "Added $add_value to the knowledge base."
            ;;
        clear)
            rm -f $HOME/.sigmakee/KBs/SUMO_NLP_KB.kif
            rm -f $HOME/.sigmakee/KBs/SUMO_NLP.kif
            rm -f $HOME/.sigmakee/KBs/SUMO.fof
            rm -f $HOME/.sigmakee/KBs/SUMO.tptp
            rm -f $HOME/.sigmakee/KBs/SUMO_NLP_QUERY.fof
            echo "Knowledge base has been cleared."
            ;;
        list)
            if [ -e "$HOME/.sigmakee/KBs/SUMO_NLP_KB.kif" ]; then
              cat $HOME/.sigmakee/KBs/SUMO_NLP_KB.kif
            fi
            ;;
        test)
            time_value=${input:6}
            cat $HOME/.sigmakee/KBs/SUMO_NLP_KB.kif > $HOME/.sigmakee/KBs/SUMO_NLP.kif
            bash prover/build_tptp.sh
            vampire --input_syntax tptp $time_value --proof tptp -qa plain --mode casc $HOME/.sigmakee/KBs/SUMO.fof
            ;;
        last)
            cat flow.txt
            ;;
        "")
            ;;
        help|man)
            printf "\n\n=================================================================\n"
            printf "\nValid commands are ask, add, clear or quit\n"
            printf '\n"ask" a question from the knowledge base.\n  Example: "ask Does CO2 cause global warming?"\n'
            printf '\n"add" will append a new sentence or file to the knowledge base.\n  Example: "add CO2 causes global warming."\n'
            printf '  Example: "add climate_facts.txt"\n'
            printf '\n"clear" will completely clear the knowledge base.\n  Example: "clear"\n'
            printf '\n"last" will show the progression through the pipeline of the last added sentence or file.\n  Example: "last"\n'
            printf '\n"list" will display the knowledge base.\n  Example: "list"\n'
            printf '\n"test" will run the Vampire prover on the knowledge base, searching for contradictions. Default is 60 seconds.\n  Example: "test -t 40" # runs for 40 seconds\n  Example: "test" # runs for 60 seconds\n'
            printf '\n"quit" will exit the interface.\n  Example: "quit"\n'
            printf "\n=================================================================\n\n"
            ;;
        quit|exit)
            echo "Good bye."
            break
            ;;
        *)
            echo "Invalid command. Please use 'ask', 'add', 'clear', 'list', 'help', or 'quit'."
            ;;
    esac
done
