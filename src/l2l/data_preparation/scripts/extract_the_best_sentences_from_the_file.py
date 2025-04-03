import json
import re
import os
import random

def load_sumo_terms(file_path):
    """Loads SUMO terms from a JSON file into a set, excluding keys that start with 'UNK'."""
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    sumo_terms = {key for key in data.keys() if not key.startswith("UNK")}
    print(f"Loaded {len(sumo_terms)} SUMO terms from {file_path} (excluding 'UNK' keys)")
    return sumo_terms

def load_sentences(file_path):
    """Loads sentences from a JSON file with 'input' (English sentence) and 'output' (SUMO logic)."""
    with open(file_path, "r") as file:
        data = json.load(file)
    print(f"Loaded {len(data)} sentences from {file_path}")
    return data

def extract_sumo_terms(sentence, sumo_terms_set):
    """Extracts SUMO terms from a sentence using regex and filters only known SUMO terms."""
    word_pattern = re.compile(r"(?<!\?)\b[a-zA-Z_][a-zA-Z_0-9]*\b")
    extracted_terms = set(word_pattern.findall(sentence))
    filtered_terms = extracted_terms.intersection(sumo_terms_set)

    # print(f"Extracted terms from sentence: {extracted_terms}")
    # print(f"Filtered SUMO terms: {filtered_terms}")

    return filtered_terms

def select_best_sentences(sentences, sumo_terms_set, max_sentences):
    """
    Selects the best sentences that maximize SUMO term coverage.
    """
    sumo_terms_dict = {}  # Maps sentences to the SUMO terms they contain
    total_sumo_terms = set()  # Tracks all unique SUMO terms found

    for item in sentences:
        extracted_terms = extract_sumo_terms(item["output"], sumo_terms_set)
        if extracted_terms:
            sumo_terms_dict[item["input"]] = extracted_terms
            total_sumo_terms.update(extracted_terms)

    print(f"Total unique SUMO terms found in sentences: {len(total_sumo_terms)}")
    print(f"Sentences with extracted SUMO terms: {len(sumo_terms_dict)}")

    selected_sentences = []
    covered_terms = set()

    while len(selected_sentences) < max_sentences:
        best_sentence = None
        best_new_terms = set()

        for sentence, terms in sumo_terms_dict.items():
            new_terms = terms - covered_terms
            if len(new_terms) > len(best_new_terms):
                best_sentence = sentence
                best_new_terms = new_terms

        if best_sentence is None:
            print("No sentence found that adds new SUMO terms. Selecting random sentences instead.")
            remaining_sentences = list(sumo_terms_dict.keys())
            random.shuffle(remaining_sentences)

            for sentence in remaining_sentences:
                if len(selected_sentences) >= max_sentences:
                    break

                selected_sentences.append({
                    "input": sentence,
                    "output": sentences[next(i for i, s in enumerate(sentences) if s["input"] == sentence)]["output"]
                })

            break

        selected_sentences.append({
            "input": best_sentence,
            "output": sentences[next(i for i, s in enumerate(sentences) if s["input"] == best_sentence)]["output"]
        })

        covered_terms.update(best_new_terms)

        del sumo_terms_dict[best_sentence]

        # print(f"Selected sentence: {best_sentence}")
        # print(f"Covered terms count: {len(covered_terms)}")
        # print(f"Remaining sentences to process: {len(sumo_terms_dict)}")

    print(f"Final selected sentences count: {len(selected_sentences)}")
    return selected_sentences


# **Usage Example**
sumo_terms_file = "/home/angelos.toutsios.gr/data/Thesis_dev/SUMO-terms/data/LatestDataSet/01-3-2025/combined_terms.json"
sentences_file = "/home/angelos.toutsios.gr/data/Thesis_dev/SUMO-terms/data/LatestDataSet/01-3-2025/combined_cleaned_from_duplicates_20_to_24_complexity.json"
output_file = os.path.splitext(sentences_file)[0]+"best_sentences_150000.json"
max_sentences = 150000

sumo_terms = load_sumo_terms(sumo_terms_file)
sentences = load_sentences(sentences_file)
best_sentences = select_best_sentences(sentences, sumo_terms, max_sentences)

# Save selected sentences
with open(output_file, "w") as file:
    json.dump(best_sentences, file, indent=4)

print(f"Selected {len(best_sentences)} sentences saved to {output_file}.")
