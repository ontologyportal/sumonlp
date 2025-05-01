import json
import random
import os

"""
Input: JSON file
Output: JSON file.

Suffles data and keep a specific number of sentences.
A better approach is the `extract_the_best_sentences_from_the_file.py` which keeps the sentences that offer the most variance!
"""

sumo_nlp_home = os.environ.get('SUMO_NLP_HOME')
l2l_home = os.path.join(sumo_nlp_home, 'src', 'l2l')
# File paths
input_json = os.path.join(l2l_home, 'data_preparation', 'data', 'input_l2l.json')

# Number of data to keep
number_of_data = 20000

# Load JSON data
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

# Shuffle the list
random.shuffle(data)

# Keep only the first 20,000 items
testing_data = data[:number_of_data]
training_data = data[number_of_data:]

training_data_number = len(data) - number_of_data
testing_data_number = number_of_data

output_training_file = os.path.splitext(input_json)[0]+"_training_"+str(training_data_number)+".json"
# Save to a new JSON file
with open(output_training_file, "w", encoding="utf-8") as f:
    json.dump(training_data, f, indent=4)


output_testing_file = os.path.splitext(input_json)[0]+"_testing_"+str(testing_data_number)+".json"
# Save to a new JSON file
with open(output_testing_file, "w", encoding="utf-8") as f:
    json.dump(testing_data, f, indent=4)

print(f"Shuffled and saved items")