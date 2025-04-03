import json
import os

file_name = "/home/angelos.toutsios.gr/data/Thesis_dev/SUMO-terms/data/LatestDataSet/01-3-2025/150k-best-selected-no-weird-detector/1-stage-training/stage1.json"
output_file = os.path.splitext(file_name)[0] + "_existing_duplicates.json"
clean_file = os.path.splitext(file_name)[0] + "_cleaned_from_duplicates.json"

def find_duplicate_inputs(json_file):
    seen = set()
    duplicates = []
    unique = []

    with open(json_file, "r") as f:
        data = json.load(f)

    for item in data:
        entry = item["input"]
        if entry in seen:
            duplicates.append(item)
        else:
            seen.add(entry)
            unique.append(item)

    print(f"Found {len(unique)} unique entries. Saving to {clean_file}")
    with open(clean_file, "w", encoding="utf-8") as f:
        json.dump(unique, f, indent=4)

    if duplicates:
        print(f"Found {len(duplicates)} exact duplicate entries. Saving to {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(duplicates, f, indent=2)
    else:
        print("No exact duplicate entries found.")


# Example usage
find_duplicate_inputs(file_name)

