#!/bin/bash

# Get the input and output file names from the command line
input_file=$1
output_file=$2

# Check if both input and output files are provided
if [ -z "$input_file" ] || [ -z "$output_file" ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

# Clear the output file if it already exists
> $output_file

# Read each line from the input file and wrap it
while IFS= read -r line; do
    echo "SUOKIF{ $line }" >> $output_file
done < $input_file

echo "Wrapping done! Check the output file: $output_file"