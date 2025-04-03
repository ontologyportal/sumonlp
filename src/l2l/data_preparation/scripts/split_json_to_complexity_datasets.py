import json
import os
import unicodedata
import re


"""
Input: JSON training data
Output: Multiple JSON files.

Based on the complexity it returns a file for each complexity category, with the related sentences.
"""


json_file = "/home/angelos.toutsios.gr/data/Thesis_dev/SUMO-terms/data/LatestDataSet/01-3-2025/combined_cleaned_from_duplicates.json"

# Load sentences from the txt file into an array (stripping whitespace)
with open(json_file, "r", encoding="utf-8") as f:
    json_data = json.load(f)  # Using a set for fast lookup

sumo_pattern = re.compile(r"(?<!\?)\b[a-zA-Z_][a-zA-Z_0-9]*\b")

filtered_data_0_to_4 = []
filtered_data_5_to_9 = []
filtered_data_10_to_14 = []
filtered_data_15_to_19 = []
filtered_data_20_to_24 = []
filtered_data_25_to_29 = []
filtered_data_30_to_34 = []
filtered_data_35_plus = []

for item in json_data:
    terms = sumo_pattern.findall(item["output"])  # Extract SUMO terms (ignoring variables)
    num_terms = len(terms)

    # Categorize based on the number of SUMO terms
    if 0 <= num_terms <= 4:
        filtered_data_0_to_4.append(item)
    elif 5 <= num_terms <= 9:
        filtered_data_5_to_9.append(item)
    elif 10 <= num_terms <= 14:
        filtered_data_10_to_14.append(item)
    elif 15 <= num_terms <= 19:
        filtered_data_15_to_19.append(item)
    elif 20 <= num_terms <= 24:
        filtered_data_20_to_24.append(item)
    elif 25 <= num_terms <= 29:
        filtered_data_25_to_29.append(item)
    elif 30 <= num_terms <= 34:
        filtered_data_30_to_34.append(item)
    elif num_terms >= 35:
        filtered_data_35_plus.append(item)


output_file = os.path.splitext(json_file)[0]

# Save filtered data to a new JSON file
with open(output_file + "_0_to_4_complexity.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data_0_to_4, f, indent=4)

with open(output_file + "_5_to_9_complexity.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data_5_to_9, f, indent=4)

with open(output_file + "_10_to_14_complexity.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data_10_to_14, f, indent=4)

with open(output_file + "_15_to_19_complexity.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data_15_to_19, f, indent=4)

with open(output_file + "_20_to_24_complexity.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data_20_to_24, f, indent=4)

with open(output_file + "_25_to_29_complexity.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data_25_to_29, f, indent=4)

with open(output_file + "_30_to_34_complexity.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data_30_to_34, f, indent=4)

with open(output_file + "_35_plus_complexity.json", "w", encoding="utf-8") as f:
    json.dump(filtered_data_35_plus, f, indent=4)