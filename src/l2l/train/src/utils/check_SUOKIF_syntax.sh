#!/bin/bash

checkSUOKIF() {
  local statement="$1"
  local saveToCSV="$2"
  echo -n "" > ".temp_output.txt"
  echo -n "" > ".error_output.txt"
  allValidStatements=true
  splitSegment=$(echo "$statement" | tr '\n' ' ')
  csvErrorOutput=""
  if [[ ! "$splitSegment" =~ ^[[:space:]]*$ ]]; then
    local input="$splitSegment"
    local length=${#input}
    local count=0
    local part=""
    local -a parts=()

    for ((i=0; i<length; i++)); do
        char=${input:$i:1}
        part+=$char
        if [[ $char == "(" ]]; then
            ((count++))
        elif [[ $char == ")" ]]; then
            ((count--))
        fi

        if [[ $count -eq 0 ]]; then
          if [[ "$part" == *"("* ]]; then
            parts+=("$part")
            part=""
          fi
        fi
    done
    # Add any residual section to the last part
    if [[ -n $part ]]; then
        if [[ ${#parts[@]} -gt 0 ]]; then
            parts[-1]+="$part"
        else
            parts+=("$part")
        fi
    fi
    # Print each part in the array
    for element in "${parts[@]}"; do
      if [[ ! "$element" =~ ^[[:space:]]*$ ]]; then
        echo -e "\n$element"
        java -Xmx8g -classpath $ONTOLOGYPORTAL_GIT/sigmakee/build/sigmakee.jar:$ONTOLOGYPORTAL_GIT/sigmakee/lib/* com.articulate.sigma.KButilities -v "$element" >> .temp_output.txt 2>>.error_output.txt
        last_line=$(tail -n 1 .temp_output.txt)
        if [ "$last_line" != "true" ]; then
          allValidStatements=false
          grepOutput=$(grep -v "VerbNet" .error_output.txt)
          echo -e "\n$grepOutput\n"
          csvErrorOutput+="$grepOutput"
        else
          echo -e "\nValid syntax\n"
        fi
      fi
    done
    
    if [ "$allValidStatements" == true ]; then
      if [ "$saveToCSV" = true ]; then
        splitSegment=$(echo "$splitSegment" | sed 's/"/""/g')
        echo "\"$splitSegment\",Valid syntax" >> "SUOKIF_Syntax_Check.csv"
      fi
    else
      if [ "$saveToCSV" = true ]; then
        splitSegment=$(echo "$splitSegment" | sed 's/"/""/g')
        csvErrorOutput=$(echo "$csvErrorOutput" | sed 's/"/""/g')
        echo "\"$splitSegment\",\"$csvErrorOutput\"" >> "SUOKIF_Syntax_Check.csv"
      fi
    fi            

    rm .temp_output.txt
    rm .error_output.txt
    echo -e "\n\n----------------------------------------------------"
  fi
}


# Check if the input file is provided as a command-line argument
if [ "$#" -ne 1 ] && [ "$#" -ne 2 ]; then
  echo -e "Usage: $0 <input-file>\n -c flag saves output to a .csv file."
  exit 1
fi

# Set the input file path from the command-line argument
inputFile="$1"

# Check if the input file exists
if [ ! -f "$inputFile" ]; then
  echo "Input file $inputFile not found."
  exit 1
fi

for arg in "$@"; do
    if [ "$arg" == "-c" ]; then
        saveToCSV=true
        echo "Saving output to SUOKIF_Syntax_Check.csv."
        echo -n "" > "SUOKIF_Syntax_Check.csv"
    fi
done

# Initialize a variable to store the current segment
currentSegment=""

# Loop through each line of the input file
while IFS= read -r line || [[ -n "$line" ]]; do
  # Check if the line is empty, only whitespace, or starts with ";" or "!!"
  if [[ -z "$line" || "$line" =~ ^[[:space:]]*$ || "$line" =~ ^[[:space:]]*\; || "$line" =~ ^\!\! ]]; then
    # Save the current segment to a variable with new lines replaced by spaces
    checkSUOKIF "$currentSegment" "$saveToCSV" 
    currentSegment=""
  else
    # Append the line to the current segment
    currentSegment+="$line"$'\n'
  fi
done < "$inputFile"

# Save and echo the last segment if there is any
if [[ -n "$currentSegment" ]]; then
  checkSUOKIF "$currentSegment" "$saveToCSV"
fi


echo "Process completed."
