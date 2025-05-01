import json
import random
import os

# Define dataset directory
sumo_nlp_home = os.environ.get('SUMO_NLP_HOME')
l2l_home = os.path.join(sumo_nlp_home, 'src', 'l2l')
# File paths
data_path = os.path.join(l2l_home, 'data_preparation', 'data')

# Define category JSON files
categories = {
    "0-4": "combined_cleaned_from_duplicates_0_to_4_complexitybest_sentences_150000",
    "5-9": "combined_cleaned_from_duplicates_5_to_9_complexitybest_sentences_150000",
    "10-14": "combined_cleaned_from_duplicates_10_to_14_complexitybest_sentences_150000",
    "15-19": "combined_cleaned_from_duplicates_15_to_19_complexitybest_sentences_150000",
    "20-24": "combined_cleaned_from_duplicates_20_to_24_complexitybest_sentences_150000",
    "25-29": "combined_cleaned_from_duplicates_25_to_29_complexitybest_sentences_150000",
    "30-34": "combined_cleaned_from_duplicates_30_to_34_complexitybest_sentences_150000",
    "35+": "combined_cleaned_from_duplicates_35_plus_complexitybest_sentences_150000"
}

# Define number of samples per stage
samples_per_stage = {
    "stage1": 1_200_000,  # 40% of total (0-9 terms)
    # "stage2": 800_000,    # 30% of total (10-24 terms)
    # "stage3": 1_000_000   # 30% of total (25+ terms)
}

# Mapping complexity ranges to training stages
stage_mapping = {
    "stage1": ["0-4", "5-9", "10-14", "15-19", "20-24", "25-29", "30-34", "35+"],
    # "stage2": ["10-14", "15-19"],
    # "stage3": ["20-24", "25-29", "30-34", "35+"]
}

# Load sentences from JSON files
def load_sentences(file):
    full_path = os.path.join(data_path, file+'.json')
    with open(full_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Select samples proportionally from multiple categories
def sample_sentences(category_list, target_size):
    all_sentences = []

    # Load and shuffle sentences from selected categories
    for category in category_list:
        file_name = categories.get(category)  # ✅ Fix: Get the correct filename
        if file_name:  # Ensure the file exists in the dictionary
            sentences = load_sentences(file_name)
            random.shuffle(sentences)  # Ensure randomness
            all_sentences.extend(sentences)

    # Shuffle and select the required amount
    random.shuffle(all_sentences)
    return all_sentences

# ✅ Fix the loop: Use `categories_list` correctly
stages = {}
for stage, categories_list in stage_mapping.items():
    stages[stage] = sample_sentences(categories_list, samples_per_stage[stage])

# Save to JSON files
for stage, sentences in stages.items():
    output_path = os.path.join(data_path, f"{stage}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sentences, f, indent=4)

print("Dataset split complete! 🚀")