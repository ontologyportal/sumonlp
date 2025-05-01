import json
import re
import os

"""
Input: JSON training data
Output: JSON training data.

After the Ollama re-word, some sentences must be processed again, and remove the \" that added
from Ollama. Also delete sentences that Ollama cannot handle.
"""
sumo_nlp_home = os.environ.get('SUMO_NLP_HOME')
l2l_home = os.path.join(sumo_nlp_home, 'src', 'l2l')
# File paths
file_name = os.path.join(l2l_home, 'data_preparation', 'data', 'input_l2l.json')

output_file = output_file = os.path.splitext(file_name)[0]+"_remove_ollama_cannot_handle.json"

# Load JSON file
with open(file_name, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = []

# Apply transformations to "input" and "output" fields
for entry in data:
    input_text = entry["input"]
    if (input_text.startswith('"') and input_text.endswith('"')):
      input_text = input_text[1:-1]
      result = {
          "input": input_text.strip(),
          "output": entry["output"]
      }
      dataset.append(result)
print(f"Initial Datase length: {len(data)}")
print(f"Final Dataset length: {len(dataset)}")
print(f"Removed sentences: {len(data) - len(dataset)}")

# Save the processed JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4)

print("Processed data saved to processed_file.json")
