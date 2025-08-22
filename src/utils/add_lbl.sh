#!/bin/bash
# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
cd ..

inputFile="$1"

# Check if argument is provided
if [ -z "$inputFile" ]; then
    echo "Error: No file provided."
    echo "Usage: $0 <file.txt>"
    exit 1
fi

# Check if the file exists and is a regular file
if [ ! -f "$inputFile" ]; then
    echo "Error: '$inputFile' does not exist or is not a regular file."
    exit 1
fi

if [[ "${inputFile: -4}" == ".txt" ]]; then
    if [ ! -f "$inputFile" ]; then
      echo "Input file $inputFile not found."
    else
      echo "Adding $inputFile to the knowledge base."
      outputFile="test/add_lbl_output.txt"
      echo "" > "$outputFile"
      line_number=0
      while IFS= read -r line; do
        ((line_number++))
        echo -e "\n\n\n\nLine $line_number: $line"
        echo "$line" > policy_extracter/input_pe.txt
        bash run_pipeline.sh
        cat prover/input_pr.txt >> "$outputFile"
        echo "!!" >> "$outputFile"
      done < "$inputFile"
    fi
    bash test/check_SUOKIF_syntax.sh $outputFile -c
    python3 test/check_SUOKIF_types.py
else
    echo "Please enter a text file name."
fi