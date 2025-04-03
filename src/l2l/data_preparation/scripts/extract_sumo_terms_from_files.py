import json
import re
import sys
from pathlib import Path
import os

"""
This script find all the SUMO terms that exist in a Json file, that it will be used for training.
It returns a Json file with all the terms and the number of encounter in all the sentences.
"""

def convert_to_json(logic_file):

    # Load JSON file
    with open(logic_file, "r") as f:
        data = json.load(f)

    # Dictionary to store unique terms and their counts
    term_dict = {}

    # Regular expression to match words while ignoring variables and numeric terms
    word_pattern = re.compile(r"(?<!\?)\b[a-zA-Z_][a-zA-Z_0-9]*\b")

    # Process each item in the JSON
    for item in data:
        tokens = word_pattern.findall(item["output"])  # Extract words, ignoring variables and numbers
        for token in tokens:
            if not token.startswith("UNK"):
                if token in term_dict:
                    term_dict[token]["count"] += 1  # Increment count if already exists
                else:
                    term_dict[token] = {"id": len(term_dict) + 1, "count": 1}  # Assign ID and start count

    output_file = os.path.splitext(logic_file)[0] + "_terms.json"


    # Save the dictionary to a JSON file
    with open(output_file, "w") as f:
        json.dump(term_dict, f, indent=4)

    print("Terms saved to terms.json")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <logic_file.txt>")
    else:
        log_file = Path(sys.argv[1])

        if not log_file.exists():
            print("Error: Log file does not exist.")
        else:
            convert_to_json(log_file)