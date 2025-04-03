import json
import re
import sys
from pathlib import Path
import os

"""
Input: JSON training data.
Output: JSON with the number of sentences that belong to each complexity category.
"""

def convert_to_json(logic_file):

    # Load JSON file
    with open(logic_file, "r") as f:
        data = json.load(f)

    # Dictionary to store unique terms and their counts

    # Regular expression to match words while ignoring variables and numeric terms
    sumo_pattern = re.compile(r"(?<!\?)\b[a-zA-Z_][a-zA-Z_0-9]*\b")

    complexity_bins = {
      "0-4 terms": 0,
      "5-9 terms": 0,
      "10-14 terms": 0,
      "15-19 terms": 0,
      "20-24 terms": 0,
      "25-29 terms": 0,
      "30-34 terms": 0,
      "35+ terms": 0
    }

    # Process each item in the JSON
    for item in data:
        terms = sumo_pattern.findall(item["output"])  # Extract SUMO terms (ignoring variables)
        num_terms = len(terms)

        # Categorize based on the number of SUMO terms
        if 0 <= num_terms <= 4:
            complexity_bins["0-4 terms"] += 1
        elif 5 <= num_terms <= 9:
            complexity_bins["5-9 terms"] += 1
        elif 10 <= num_terms <= 14:
            complexity_bins["10-14 terms"] += 1
        elif 15 <= num_terms <= 19:
            complexity_bins["15-19 terms"] += 1
        elif 20 <= num_terms <= 24:
            complexity_bins["20-24 terms"] += 1
        elif 25 <= num_terms <= 29:
            complexity_bins["25-29 terms"] += 1
        elif 30 <= num_terms <= 34:
            complexity_bins["30-34 terms"] += 1
        elif num_terms >= 35:
            complexity_bins["35+ terms"] += 1


        output_file = os.path.splitext(logic_file)[0] + "_complexity.json"


    # Save the dictionary to a JSON file
    with open(output_file, "w") as f:
        json.dump(complexity_bins, f, indent=4)

    print("Complexity saved.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <logic_file.txt>")
    else:
        log_file = Path(sys.argv[1])

        if not log_file.exists():
            print("Error: Log file does not exist.")
        else:
            convert_to_json(log_file)