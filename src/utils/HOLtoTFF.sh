#!/bin/bash
# Gets the path to this script, so this script can be run from any location and still work
parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" || exit; pwd -P )
cd "$parent_path" || exit
cd ..

input_file="$1"
temp_file=$(mktemp)

# Removes all temporal relations from the formula.
# Take care as this can change the semantic meaning
# and result in a semantically incorrect answer.
echo "Removing temporal relations"
while IFS= read -r line; do
    echo "Before removal: $line"
    result=$(java -Xmx8g -classpath "$SIGMA_SRC/build/sigmakee.jar:$SIGMA_SRC/lib/*" \
        com.articulate.sigma.Formula -r "$line" | tail -n 3 | head -n 1)
    echo "After removal: $result"
    echo "$result" >> "$temp_file"
done < "$input_file"


mv "$temp_file" "$1"
