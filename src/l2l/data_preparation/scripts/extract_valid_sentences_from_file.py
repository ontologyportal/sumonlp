import json
import os
import unicodedata
import re

"""
This scripts takes as input a json file with the Valid sentences after using the Weirdness Detector
and the Json file will all the input/output pairs.

It returns a JSON file with only the VALID sentences with the corresponding output.
"""

# Define file paths
filtered_json_file = "/home/angelos.toutsios.gr/data/Thesis_dev/SUMO-terms/data/LatestDataSet/new_dataset/combined_valid_sentences_only.json"
json_file = "/home/angelos.toutsios.gr/data/Thesis_dev/SUMO-terms/data/LatestDataSet/new_dataset/combined.json"

# Function to normalize text and remove hidden characters
def clean_text(text):
    text = unicodedata.normalize("NFKC", text)  # Normalize Unicode
    text = re.sub(r"[^\S\n]+", " ", text)  # Replace multiple spaces, tabs with a single space
    text = text.strip()  # Remove leading/trailing spaces
    return text

# Load sentences from the txt file into an array (stripping whitespace)
with open(filtered_json_file, "r", encoding="utf-8") as f:
    json_sentences = json.load(f)  # Using a set for fast lookup

sentence_list = set()

for entry in json_sentences:
  sentence_list.add(entry["input"])

print(f"Number of Valid sentences: {len(sentence_list)}")


# Load JSON data
with open(json_file, "r", encoding="utf-8") as f:
    json_data = json.load(f)

filtered_data = []
seen_sentences = set()  # Track already added sentences

for entry in json_data:
    # cleaned_input = clean_text(entry["input"])  # Normalize sentence
    cleaned_input = entry["input"]

    if cleaned_input in sentence_list and cleaned_input not in seen_sentences:
        filtered_data.append(entry)
        seen_sentences.add(cleaned_input)

# Print total matches found
print(f"Total matches found: {len(filtered_data)}")

output_file = os.path.splitext(json_file)[0] + "_filtered.json"

# Save filtered data to a new JSON file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_data, f, indent=4)
