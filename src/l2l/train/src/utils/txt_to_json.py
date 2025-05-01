import sys
import json
import time
from pathlib import Path
import os

"""
Input: .txt file with Input sentences
       .txt file with Logic sentences

Output: Json file with Input/Label objects.
"""

def convert_to_json(english_file, logic_file):
    try:
        with open(english_file, 'r', encoding='utf-8') as eng_f, open(logic_file, 'r', encoding='utf-8') as log_f:
            english_lines = eng_f.readlines()
            logic_lines = log_f.readlines()

        if len(english_lines) != len(logic_lines):
            print("Warning: The number of lines in the files do not match. Some data may be missing.")

        dataset = []
        for eng, log in zip(english_lines, logic_lines):
            dataset.append({
                "input": eng.strip(),
                "output": log.strip()
            })

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_filename = f"sumo_dataset_{timestamp}.json"

        with open(output_filename, 'w', encoding='utf-8') as json_f:
            json.dump(dataset, json_f, indent=4, ensure_ascii=False)

        print(f"Dataset saved to {output_filename}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <english_file.txt> <logic_file.txt>")
    else:

        eng_file = Path(sys.argv[1])
        log_file = Path(sys.argv[2])

        print(f"Current working directory: {os.getcwd()}")
        print(f"eng_file: {eng_file}")
        print(f"log_file: {log_file}")
        print(f"eng_file.exists(): {eng_file.exists()}")
        print(f"log_file.exists(): {log_file.exists()}")

        if not eng_file.exists() or not log_file.exists():
            print("Error: One or both input files do not exist.")
        else:
            convert_to_json(eng_file, log_file)