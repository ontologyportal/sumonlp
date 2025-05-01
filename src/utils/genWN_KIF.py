#!/usr/bin/env python3
#
# Takes all wordnet subsuming mappings, and auto-generates a .kif file with
# associated subclass and documentation statements.
#



import os
import glob
import sys
import unicodedata
import re
from KB_reader import KB_reader

created_terms = set()
# Dictionary to convert numeric digits to their spelled-out form
number_words = {
    '0': 'Zero',
    '1': 'One',
    '2': 'Two',
    '3': 'Three',
    '4': 'Four',
    '5': 'Five',
    '6': 'Six',
    '7': 'Seven',
    '8': 'Eight',
    '9': 'Nine'
}


def extract_mappings(line):
    # Match &% followed by content, ending with either + or = (capture both content and delimiter)
    matches = re.findall(r"&%([^+=]+)([+=])", line)
    # Keep only those where the delimiter is +
    mappings = [m[0].strip() for m in matches if m[1] == '+']
    if not mappings:
        if line.strip().endswith('+'):
            print(f"No subsuming mapping for line but ends in '+' for line: {line.strip()}")
        return mappings
    mappings = [item for item in mappings if item not in relations]
    return mappings if mappings else None


def extract_documentation(line):
    documentation = ""
    try:
        start = line.index("|") + 1
        end = line.find("&%", start)  # Find the first "&%" after "|"
        if start < end:
            documentation = line[start:end].strip()
            documentation = documentation.replace('"', "'")  # Replace double quotes with single
    except ValueError:
        return None
    return documentation


def extract_synset(line):
    words = line.split()
    if len(words) < 4:
        return None

    try:
        n = 2*int(words[3], 16)
        synset = words[4:4 + n]
        synset = synset[::2]
    except (IndexError, ValueError):
        return None
    return synset


def camel_case_word(word):
    # Replace hyphens/underscores and capitalize the letter following each
    word = re.sub(r'[-_](\w)', lambda m: m.group(1).upper(), word)
    return re.sub(r'[^a-zA-Z0-9]', '', word)


def create_new_term(file_name, synset, mappings):
    newTerm = camel_case_word(synset[0]) + mappings[0]
    newTerm = newTerm[0].upper() + newTerm[1:]

    # If the term starts with digits, replace all leading digits with spelled-out forms
    if newTerm and newTerm[0].isdigit():
        # Find how many consecutive digits are at the start
        i = 0
        while i < len(newTerm) and newTerm[i].isdigit():
            i += 1
        # Replace each digit with its word form
        spelled_prefix = ''
        for digit in newTerm[:i]:
            spelled_prefix += number_words[digit]

        # Combine the spelled prefix with the rest of the term
        newTerm = spelled_prefix + newTerm[i:]

    original_term = newTerm
    counter = 1
    while newTerm in created_terms:
        newTerm = f"{original_term}{counter}"
        counter += 1
    created_terms.add(newTerm)
    return newTerm


def write_to_file(out_f, newTerm, mappings, documentation, synset):
    global reader
    out_f.write(f"\n\n;; ;;;;;;;;;;;;;;;;;;;; {newTerm} ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;\n")
    out_f.write(f"(documentation {newTerm} EnglishLanguage \"{documentation}\")\n")
    for mapping in mappings:
        if mapping in attributes:
            if reader.isInstance(mapping):
                out_f.write(f"(subAttribute {newTerm} {mapping})\n")
                #print ("\n\n*********** " + newTerm + " " + mapping + " *******************")
                #print ("(subAttribute " + newTerm + " " + mapping+")")
            else:
                out_f.write(f"(subclass {newTerm} {mapping})\n")
                #print ("\n\n*********** " + newTerm + " " + mapping + " *******************")
                #print ("(subclass " + newTerm + " " + mapping+")")

        else:
            out_f.write(f"(subclass {newTerm} {mapping})\n")

    for synset_element in synset:
        synset_element = synset_element.replace("_", " ")
        out_f.write(f"(termFormat EnglishLanguage {newTerm} \"{synset_element}\")\n")

# SUMO is utf-8, but wordnet is not. We need to handle words with special characters.
# normalize tries to replace characters, like accented e, with regular e.
def clean_text(text):
    # Normalize and remove accents
    text = ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )
    # Remove the Unicode replacement character
    return text.replace('�', '')

def process_file(file_path, out_f):
    file_name = os.path.basename(file_path)
    found_count = 0
    try:
        with open(file_path, 'r', encoding='windows-1252') as in_f:
            for line_num, line in enumerate(in_f, 1):
                if line.lstrip().startswith(';'): # skip comments
                    continue

                if line.strip().endswith(('+', '=')): # sometimes there is more than one mapping. need to check.
                    line = clean_text(line)
                    mappings = extract_mappings(line)
                    if not mappings:
                        continue

                    documentation = extract_documentation(line)
                    if not documentation:
                        print(f"No documentation found for line: {line.strip()}")
                        continue

                    synset = extract_synset(line)
                    if not synset:
                        print(f"Empty synset for line: {line.strip()}")
                        continue

                    newTerm = create_new_term(file_name, synset, mappings)

                    write_to_file(out_f, newTerm, mappings, documentation, synset)
                    found_count += 1
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")
        return 0
    return found_count


def find_subsuming_mappings(directory_path, output_file):
    if '$' in directory_path:
        directory_path = os.path.expandvars(directory_path)

    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
        sys.exit(1)

    file_pattern = os.path.join(directory_path, "WordNetMappings30*.txt")
    text_files = glob.glob(file_pattern)

    if not text_files:
        print(f"No text files found in '{directory_path}'.")
        sys.exit(1)

    found_count = 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for file_path in sorted(text_files):
            found_count += process_file(file_path, out_f)

    print(f"Search complete. Found {found_count} subsuming mappings in {len(text_files)} files.")
    print(f"Results saved to '{output_file}'.")


if __name__ == "__main__":
    reader = KB_reader()
    attributes = reader.getAllSubClassesSubAttributesInstances("Attribute")
    relations = reader.getAllSubClassesSubAttributesInstances("Relation")

    search_directory =  os.path.expandvars("$ONTOLOGYPORTAL_GIT/sumo/WordNetMappings/")
    output_file_path =  os.path.expandvars("$ONTOLOGYPORTAL_GIT/sumo/WN_Subsuming_Mappings.kif")
    find_subsuming_mappings(search_directory, output_file_path)



