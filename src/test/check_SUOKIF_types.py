import os
import re
import csv
import sys
from collections import defaultdict

# Clean line but preserve ? characters
def clean_line(line):
    line_no_quotes = re.sub(r'"[^"]*"', '', line)
    line_no_parens = re.sub(r'[()]', '', line_no_quotes)
    line_cleaned = re.sub(r'[^\w\s?]', '', line_no_parens)
    return line_cleaned

# Extract words, discard ignored and ?-prefixed ones
def extract_words(line):
    ignored_words = {"exists", "and", "or", "instance", "now" }
    words = re.findall(r'\??\b\w+\b', line)
    return [
        word for word in words
        if not word.startswith('?') and word.lower() not in ignored_words
    ]

# Process all .kif files and store words (not in quotes or comments) with a file location
def process_kif_files(kif_dir):
    word_locations = defaultdict(set)
    print("Looking for kif files in " + kif_dir)
    for root, _, files in os.walk(kif_dir):
        for file in files:
            if file.endswith(".kif"):
                print ("Processing: " + file)
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                    # Remove multi-line quoted strings
                    content_no_quotes = re.sub(r'"(?:[^"\\]|\\.|\\\n)*?"', '', content, flags=re.DOTALL)

                    # Process line-by-line
                    for line in content_no_quotes.splitlines():
                        line = line.strip()
                        if not line or line.startswith(";"):
                            continue
                        cleaned_line = re.sub(r'[^\w\s?]', '', line)
                        words = re.findall(r'\b\w+\b', cleaned_line)
                        for word in words:
                            if not word.startswith('?'):
                                word_locations[word].add(file_path)

    return word_locations

# Check phrase against word_locations, return result_str, not_found_str, instance_str
def check_phrase_in_word_locations(phrase, word_locations):
    result = []
    not_found_words = []
    instance_words = []

    cleaned_phrase = clean_line(phrase)
    phrase_words = extract_words(cleaned_phrase)

    for word in phrase_words:
        if word in word_locations and word_locations[word]:
            first_location = sorted(word_locations[word])[0]
            result.append(f"Word: '{word}' found in {first_location}")
        else:
            not_found_words.append(word)
            result.append(f"Word: '{word}' not found in any .kif files.")
            if f"instance {word}" in phrase:
                instance_words.append(word)

    result_str = " | ".join(result)
    not_found_str = ", ".join(not_found_words) if not_found_words else "None"
    instance_str = ", ".join(instance_words) if instance_words else "None"

    return result_str, not_found_str, instance_str

# Process the CSV: preserve column 2, add match results in columns 3–5
def process_csv(input_csv, output_csv, word_locations):
    rows = []

    with open(input_csv, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row:
                phrase = row[0]
                print(f"Checking phrase: '{phrase}'")

                result_str, not_found_str, instance_str = check_phrase_in_word_locations(phrase, word_locations)
                updated_row = row[:2] + [result_str, instance_str, not_found_str]
                rows.append(updated_row)

    with open(output_csv, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(rows)

# Main execution
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input_file.csv> <output_file.csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    kif_dir = os.path.expanduser("~/.sigmakee/KBs")


    print("Processing .kif files...")
    word_locations = process_kif_files(kif_dir)

    print("Processing CSV file...")
    process_csv(input_csv, output_csv, word_locations)

    print(f"Done. Results saved to '{output_csv}'")
