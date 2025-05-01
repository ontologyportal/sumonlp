import json
import re
import os

"""
Input: JSON file with training data
Output: JSON file

Removes spaces between ? and Variable Names. Also replaces the var names with X<number>.

"""


sumo_nlp_home = os.environ.get('SUMO_NLP_HOME')
l2l_home = os.path.join(sumo_nlp_home, 'src', 'l2l')
# File paths
file_name = os.path.join(l2l_home, 'data_preparation', 'data', 'input_l2l.json')

output_file = output_file = os.path.splitext(file_name)[0]+"_var_replaced.json"


# Load JSON file
with open(file_name, "r", encoding="utf-8") as f:
    data = json.load(f)

# Function to clean text (remove space after ?)
def clean_text(text):
    return re.sub(r'\? +', '?', text)  # Replace "? " with "?"

# Function to replace variables
def replace_variables(text):
    variables = re.findall(r'\?\w+', text)  # Find all words starting with ?
    unique_vars = list(dict.fromkeys(variables))  # Remove duplicates while preserving order

    replacement_map = {var: f"?X{i+1}" for i, var in enumerate(unique_vars)}  # Create mapping

    # Replace variables in text
    for var, new_var in replacement_map.items():
        text = text.replace(var, new_var)

    return text

# Apply transformations to "input" and "output" fields
for entry in data:
    entry["input"] = entry["input"]
    entry["output"] = clean_text(entry["output"])  # Remove space after ?
    entry["output"] = replace_variables(entry["output"])  # Replace variables

# Save the processed JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Processed data saved to processed_file.json")
