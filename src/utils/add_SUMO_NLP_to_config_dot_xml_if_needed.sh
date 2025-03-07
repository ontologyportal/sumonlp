#!/bin/bash

file="$SIGMA_HOME/KBs/config.xml"

# Define the line to check for
line_to_check='<constituent filename="SUMO_NLP.kif" />'

# Define the line to be replaced and the new line to insert
line_to_replace='</kb>'
new_line='<constituent filename="SUMO_NLP.kif" />\n</kb>'

# Check if the line is present in the file
if grep -q "$line_to_check" "$file"; then
    echo "'$line_to_check' already exists in $file."
else
    # Replace the line
    sed -i "s|$line_to_replace|$new_line|" "$file"
    echo "SUMO_NLP.kif did not exist in $file and has been added."
fi
