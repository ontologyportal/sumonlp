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

from itertools import islice

created_terms = set()
roots_and_kids = {}
new_instances = set()
new_classes_synset_id = {}

search_directory =  os.path.expandvars("$ONTOLOGYPORTAL_GIT/sumo/WordNetMappings/")
output_file_path =  os.path.expandvars("$ONTOLOGYPORTAL_GIT/sumo/wn_kif_files")

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

def extract_synset_id(line):
    parts = line.strip().split()
    if len(parts) < 3:
        return None
    offset = parts[0]
    pos = parts[2]
    return f"{offset}-{pos}"

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

def extract_hypernyms_from_line(line):
    parts = line.strip().split()
    if len(parts) < 4:
        return []

    # Get number of words in the synset (hex)
    word_count = int(parts[3], 16)
    lemma_start = 4
    lemma_end = lemma_start + (2 * word_count)
    pointer_count_index = lemma_end

    pointer_count = int(parts[pointer_count_index])
    pointer_start = pointer_count_index + 1

    hypernym_ids = []
    for i in range(pointer_count):
        pointer_index = pointer_start + i * 4
        symbol = parts[pointer_index]
        if symbol == "@":  # hypernym
            offset = parts[pointer_index + 1]
            pos = parts[pointer_index + 2]
            hypernym_ids.append(f"{offset}-{pos}")
    return hypernym_ids



def camel_case_word(word):
    # Replace hyphens/underscores and capitalize the letter following each
    word = re.sub(r'[-_](\w)', lambda m: m.group(1).upper(), word)
    return re.sub(r'[^a-zA-Z0-9]', '', word)


def create_new_term(synset, parent, grandparent):
    if grandparent:
        parent = ''.join(parent.rsplit(grandparent, 1)) # Remove the grandparent term from the list, so that we don't have long term names like RobinBirdAnimalOrganismPhysicalEntity. We just have RobinBird
    newTerm = camel_case_word(synset[0]) + parent
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
    while newTerm in created_terms: # Sometimes two terms happen to have the same name, synonyms, and PoS.
        newTerm = f"{original_term}{counter}"
        counter += 1
    created_terms.add(newTerm)
    return newTerm


def write_to_file(outputfile, newTerm, mappings, documentation, synset):
    global reader
    out_f = open(outputfile, 'a')
    out_f.write(f"\n\n;; ;;;;;;;;;;;;;;;;;;;; {newTerm} ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;\n")
    out_f.write(f"(documentation {newTerm} EnglishLanguage \"{documentation}\")\n")
    for mapping in mappings:
        if mapping in attributes:
            if newTerm in attributes:
                if reader.isInstance(mapping) or mapping in new_instances:
                    out_f.write(f"(subAttribute {newTerm} {mapping})\n")
                else:
                    out_f.write(f"(instance {newTerm} {mapping})\n")
                    new_instances.add(newTerm)
            else:
                out_f.write(f"(attribute {newTerm} {mapping})\n")
        else:
            out_f.write(f"(subclass {newTerm} {mapping})\n")

    for synset_element in synset:
        synset_element = synset_element.replace("_", " ")
        out_f.write(f"(termFormat EnglishLanguage {newTerm} \"{synset_element}\")\n")
    out_f.close()

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

def process_file(file_path):
    found_count = 0
    try:
        with open(file_path, 'r', encoding='windows-1252') as in_f:
            for line_num, line in enumerate(in_f, 1):
                if line.lstrip().startswith(';'): # skip comments
                    continue

                if line.strip().endswith(('+', '=')): # sometimes there is more than one mapping. need to check.
                    line = clean_text(line)

                    synset_id = extract_synset_id(line)
                    if not synset_id:
                        print(f"No synset id found for line: {line.strip()}")
                        continue

                    synset = extract_synset(line) # If there is no term in the synset, this is an error.
                    if not synset:
                        print(f"Empty synset for line: {line.strip()}")
                        continue

                    documentation = extract_documentation(line) #don't add lines
                    if not documentation:
                        print(f"No documentation found for line: {line.strip()}")
                        continue

                    mappings = extract_mappings(line)
                    if not mappings: # If there is no subsuming mapping, then skip it
                        continue
                    for mapping in mappings: # All conditions have been met, put it in the map for future processing.
                        roots_and_kids.setdefault(mapping, {})[synset_id] = line
                    found_count += 1
    except Exception as e:
        print(f"Error processing file '{file_path}': {e}")
        return 0
    return found_count

global numSubmappedWords
def process_term(root, children_map, synset_id, wn_line):
    mappings = extract_mappings(wn_line) # These are defined in SUMO
    hypernyms = extract_hypernyms_from_line(wn_line) # These are defined in wordnet.
    # Add hypernyms to mappings, but only if they are a child of the root of this branch of the ontology (so as not to conflict with SUMO defined mappings)
    parent = ""
    grandparent = ""
    for hypernym in hypernyms:
        if hypernym in children_map:
            if root in mappings:
                mappings.remove(root) # We remove the link directly to the parent and replace it with a sub-node.
            if not children_map[hypernym].startswith("PROCESSED"):
                process_term(root, children_map, hypernym, children_map[hypernym])
            parent = children_map[hypernym].split(":::")[1]
            grandparent = children_map[hypernym].split(":::")[2]
            mappings.insert(0, parent) # Hypernyms go in front, makes naming convention more meaningful.

    synset = extract_synset(wn_line)
    newTerm = create_new_term(synset, parent, grandparent)
    if not any(mapping not in attributes for mapping in mappings):
        attributes.add(newTerm)
    documentation = extract_documentation(wn_line)
    # If a subsuming mapping has more than 100 children terms, it gets its own .kif file.
    # Otherwise, it gets thrown in the UNCATEGORIZED.kif file.
    filename = output_file_path + "/" + root + ".kif"
    if len(children_map) < 100:
        filename = output_file_path + "/UNCATEGORIZED.kif"
    write_to_file(filename, newTerm, mappings, documentation, synset)
    children_map[synset_id] = "PROCESSED:::"+newTerm+":::"+mappings[0] # synset_id -> PROCESSED:::newTerm:::parent
    global numSubmappedWords
    for mapping in mappings:
        if mapping != root:
            print(f"child: {newTerm} maps to: {mapping} instead of {root}")
            numSubmappedWords += 1



def generate_new_kifs():
    # roots and kids is of the form: {SubsumingMapping: {synset_id: Word Net line}}
    global numSubmappedWords
    numSubmappedWords = 0
    for root, children_map in roots_and_kids.items():
        for synset_id, wn_line in children_map.items():
            if not wn_line.startswith("PROCESSED"):
                process_term(root, children_map, synset_id, wn_line)
    print("Total number of reconfigured hyponyms: " + str(numSubmappedWords))
    total = 0
    for children_map in roots_and_kids.values():
        total += len(children_map)
    print("Total terms processed: " + str(total))


def find_subsuming_mappings():
    global search_directory
    if '$' in search_directory:
        search_directory = os.path.expandvars(search_directory)

    if not os.path.isdir(search_directory):
        print(f"Error: Directory '{search_directory}' not found.")
        sys.exit(1)

    file_pattern = os.path.join(search_directory, "WordNetMappings30*.txt")
    text_files = glob.glob(file_pattern)

    if not text_files:
        print(f"No text files found in '{search_directory}'.")
        sys.exit(1)

    found_count = 0
    for file_path in sorted(text_files):
        found_count += process_file(file_path)
    print(f"Search complete. Found {found_count} subsuming mappings in {len(text_files)} files.")

    print("Generating new knowledge bases ...")
    generate_new_kifs()
    print(f"Results saved to " + output_file_path)


if __name__ == "__main__":
    reader = KB_reader()
    attributes = reader.getAllSubClassesSubAttributesInstancesOf("Attribute")
    relations = reader.getAllSubClassesSubAttributesInstancesOf("Relation")

    os.makedirs(output_file_path, exist_ok=True)
    find_subsuming_mappings()



