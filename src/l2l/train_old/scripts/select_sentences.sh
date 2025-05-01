#!/bin/bash

# Set the default percentages
DEFAULT_TRAIN_PERCENTAGE=90
DEFAULT_VAL_PERCENTAGE=10

# Set the input files
input_file="/home/angelos.toutsios.gr/workspace/sumonlp/data/combined-eng.txt-0"
output_file="/home/angelos.toutsios.gr/workspace/sumonlp/data/combined-log.txt-0"

# Check if an argument is provided
if [ "$#" -eq 1 ]; then
    # Validate that the argument is a positive integer
    if ! [[ "$1" =~ ^[0-9]+$ ]] || [ "$1" -le 0 ]; then
        echo "Error: Please provide a valid positive integer for the number of training sentences."
        exit 1
    fi
    train_samples=$1
    val_samples=$(( train_samples / 9 ))  # 10% of training samples
else

    # Count total lines in the input files
    total_lines=$(wc -l < "$input_file")
    echo "Total lines: $total_lines"

    # Calculate training and validation sizes
    train_samples=$(( total_lines * DEFAULT_TRAIN_PERCENTAGE / 100 ))
    val_samples=$(( total_lines * DEFAULT_VAL_PERCENTAGE / 100 ))
fi

# Paths to new files
train_input_file="input_sentences.txt"  # New input file
train_output_file="output_logical.txt"  # New output file

val_input_file="input_sentences_val.txt"
val_output_file="output_logical_val.txt"

# Step 1: Combine lines from both files to keep sentence pairs aligned
temp_combined_file="combined_temp.txt"
paste "$input_file" "$output_file" > "$temp_combined_file"

# Step 2: Shuffle the combined pairs
shuffled_combined_file="shuffled_combined.txt"
shuf "$temp_combined_file" > "$shuffled_combined_file"

# Step 3: Create training data from the first X shuffled pairs
head -n "$train_samples" "$shuffled_combined_file" > "train_combined_temp.txt"

# Step 4: Create validation data from the next All-X shuffled pairs
# Calculate the ending line for the validation samples
val_end=$(( train_samples + val_samples ))
sed -n "$(( train_samples + 1 )),$val_end p" "$shuffled_combined_file" > "val_combined_temp.txt"

# Step 5: Separate combined pairs back into individual files for training and validation
# For training
cut -f1 "train_combined_temp.txt" > "$train_input_file"    # Input sentences
cut -f2 "train_combined_temp.txt" > "$train_output_file"   # Corresponding logical forms

# For validation
cut -f1 "val_combined_temp.txt" > "$val_input_file"        # Input sentences
cut -f2 "val_combined_temp.txt" > "$val_output_file"       # Corresponding logical forms

# Clean up temporary files
rm "$temp_combined_file" "$shuffled_combined_file" "train_combined_temp.txt" "val_combined_temp.txt"

# Notify the user
echo "New shuffled training files created: $train_input_file, $train_output_file"
echo "New shuffled validation files created: $val_input_file, $val_output_file"
