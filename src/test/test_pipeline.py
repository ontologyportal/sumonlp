import os
import csv
import subprocess

# Set the working directory to the script's location
parent_path = os.path.dirname(os.path.abspath(__file__))
os.chdir(parent_path)

# Start any needed services
subprocess.run(["bash", "../utils/start_ollama.sh"], check=True)

# Define input and output CSV file names
INPUT_CSV = "sentences.csv"
RESULTS_CSV = "pipeline_test_results.csv"

# Define the pipeline stages with their input/output files and expected CSV column names.
stages = [
    {
        "name": "metaphor_handling",
        "input_file": "../metaphor_handling/input_mh.txt",
        "entry": "../metaphor_handling/entry_point.sh",
        "output_file": "../metaphor_handling/output_mh.txt",
        "expected_field": "expected_metaphor",
        "actual_field": "actual_metaphor",
        "status_field": "metaphor_status"
    },
    {
        "name": "sentence_simplification",
        "input_file": "../sentence_simplification/input_ss.txt",
        "entry": "../sentence_simplification/entry_point.sh",
        "output_file": "../sentence_simplification/output_ss.txt",
        "expected_field": "expected_sentence_simplification",
        "actual_field": "actual_sentence_simplification",
        "status_field": "simplification_status"
    },
    {
        "name": "oov_handling",
        "input_file": "../oov_handling/input_oov.txt",
        "entry": "../oov_handling/entry_point.sh",
        "output_file": "../oov_handling/output_oov.txt",
        "expected_field": "expected_oov",
        "actual_field": "actual_oov",
        "status_field": "oov_status"
    },
    {
        "name": "l2l",
        "input_file": "../l2l/input_l2l.txt",
        "entry": "../l2l/entry_point.sh",
        "output_file": "../l2l/output_l2l.txt",
        "expected_field": "expected_l2l",
        "actual_field": "actual_l2l",
        "status_field": "l2l_status"
    },
    {
        "name": "oov_postprocessing",
        "input_file": "../oov_handling/input_post_oov.txt",
        "entry": "../oov_handling/entry_point_postprocessing.sh",
        "output_file": "../oov_handling/output_post_oov.txt",
        "expected_field": "expected_post_oov",
        "actual_field": "actual_post_oov",
        "status_field": "post_oov_status"
    }
]

# The mapping for what to use as input for each stage.
# Stage 1 uses the original sentence, and each subsequent stage uses the expected output from the previous stage.
input_for_stage = [
    "sentence",            # Stage 1 input: the original sentence.
    "expected_metaphor",   # Stage 2 input: expected output from stage 1.
    "expected_sentence_simplification",  # Stage 3 input: expected output from stage 2.
    "expected_oov",        # Stage 4 input: expected output from stage 3.
    "expected_l2l"         # Stage 5 input: expected output from stage 4.
]

# Prepare results header
results_header = [
    "sentence",
    "expected_metaphor", "actual_metaphor", "metaphor_status",
    "expected_sentence_simplification", "actual_sentence_simplification", "simplification_status",
    "expected_oov", "actual_oov", "oov_status",
    "expected_l2l", "actual_l2l", "l2l_status",
    "expected_post_oov", "actual_post_oov", "post_oov_status"
]

results = []  # List to hold result rows

# Open and parse the input CSV using Python's csv module
with open(INPUT_CSV, newline="", encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # Create a dictionary to store this test case's results.
        res_row = {}
        res_row["sentence"] = row["sentence"]
        
        # Loop through each stage and run the test.
        for idx, stage in enumerate(stages):
            # For stage 1, use the original sentence; for subsequent stages, use the expected value from CSV.
            input_field = input_for_stage[idx]
            stage_input = row[input_field]

            # Write the stage input to its input file.
            with open(stage["input_file"], "w", encoding="utf-8") as f:
                f.write(stage_input)

            # Execute the stage.
            subprocess.run(["bash", stage["entry"]], check=True)

            # Read the output from the stage.
            with open(stage["output_file"], "r", encoding="utf-8") as f:
                # Remove trailing whitespace/newlines
                actual_output = f.read().strip()

            # Get the expected output from CSV.
            expected_output = row[stage["expected_field"]].strip()

            # Compare the actual output to the expected output.
            status = "pass" if actual_output == expected_output else "fail"

            # Save the results for this stage.
            # For naming consistency, we use keys like "actual_policy", etc.
            res_row[stage["expected_field"]] = expected_output
            res_row[stage["actual_field"]] = actual_output
            res_row[stage["status_field"]] = status

        # Append the result for this sentence to our list.
        results.append(res_row)

# Write the results to a CSV file.
with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=results_header, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for res_row in results:
        writer.writerow(res_row)

print(f"Pipeline testing complete. Results saved to {RESULTS_CSV}")
