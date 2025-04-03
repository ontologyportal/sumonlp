import re

"""
This script extracts the SUMO terms that exist in a .kif file!
"""

# Input and output file names
input_file = "mega.kif"  # Change to your actual KIF file
output_file = "sumo_terms.txt"

# Regex pattern to capture the word before the double quotes
pattern = r'\(termFormat EnglishLanguage (\S+) ".*"\)'

# Extracted words array
words = []

# Read the KIF file and extract words
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        match = re.match(pattern, line)
        if match:
            word = match.group(1)
            if word not in words:
              words.append(word)  # Capture the word before quotes

# Save words to output file
with open(output_file, "w", encoding="utf-8") as f:
    f.write("\n".join(words) + "\n")

# Read and print the saved words
with open(output_file, "r", encoding="utf-8") as f:
    saved_words = [line.strip() for line in f]

print("Sumo Term Extraction completed!")