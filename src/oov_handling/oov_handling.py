import sqlite3
import stanza
import os
import time

# Initialize Stanza pipeline for English
stanza.download('en', processors='tokenize,pos,lemma', verbose=False)  # Download model if not already done
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma', verbose=False)

# Database configuration
DB_PATH = 'vocabulary.db'

# Connect to the SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Ensure the Dictionary and UnknownWords tables exist
cursor.execute("""
CREATE TABLE IF NOT EXISTS UnknownWords (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT UNIQUE,
    used INTEGER DEFAULT 0 CHECK (used IN (0, 1))
)
""")
conn.commit()

def get_word_type(word):
    """Determine if the word is a noun or verb using Stanza."""
    if word.upos == 'NOUN':
        return 'noun'
    elif word.upos == 'PROPN':
      return 'noun-phrase'
    elif word.upos == 'VERB':
        return 'verb'
    else:
        return None

def check_word_in_dictionary(root, word_type):
    """Check if the word exists in the Dictionary table."""
    cursor.execute("SELECT id FROM Word WHERE root = ? AND pos = ?", (root.lower(), word_type))
    # cursor.execute("SELECT id FROM Dictionary WHERE root = ?", (root.lower()))
    return cursor.fetchone()

def add_unknown_word(word, root, word_type):
    """Check if an unknown word already exists in the UnknownWords table with 'used' set to 0, or insert it if not."""
    try:
        # trim word
        word = word.strip()
        word = word.lower()
        cursor.execute("SELECT * FROM UnknownWords WHERE word = ? AND used = 0", (word,))
        result = cursor.fetchone()
        if result:
            return result[0] # Return the ID if the word exists
        else:
            cursor.execute("INSERT INTO UnknownWords (word) VALUES (?)", (word,))
            conn.commit()
            return cursor.lastrowid # Return the ID of the newly inserted word
    except sqlite3.IntegrityError as e:
        print(f"Error: {e}")
        return None


def process_sentence(sentence):
    """Process a single sentence, replacing unknown nouns and verbs with tags."""
    doc = nlp(sentence)
    processed_tokens = []

    for sent in doc.sentences:
        for word in sent.words:
            word_type = get_word_type(word)
            if word_type:

                # Check if the root form exists in the Dictionary
                dictionary_entry = check_word_in_dictionary(word.lemma, word_type)
                if dictionary_entry:
                    # Known word, keep the original token text
                    processed_tokens.append(word.text)
                else:
                    # Unknown word, replace with <UNK_type_id>
                    unk_id = add_unknown_word(word.text , word.lemma, word_type)
                    if unk_id != None:
                        processed_tokens.append(f"<UNK_{word_type}_{unk_id}>")
                    else:
                        # In case of any issue, keep the original word
                        processed_tokens.append(word.text)
            else:
                # Non-noun/verb word, keep as-is
                processed_tokens.append(word.text)

    # Join processed tokens to form the sentence with appropriate spacing
    return ' '.join(processed_tokens)

def process_file(input_file, output_file):
    """Process the input file and write the processed text to the output file."""
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            stripped_line = line.strip()
            if stripped_line:
                processed_line = process_sentence(stripped_line)
                outfile.write(processed_line + '\n')
            else:
                outfile.write('\n')  # Preserve empty lines



if __name__ == "__main__":
    start_time = time.time()
    input_filename = 'input_oov.txt'    # Input file containing sentences
    output_filename = 'output_oov.txt'  # Output file to save processed sentences

    process_file(input_filename, output_filename)

    # Close the database connection after processing
    conn.close()
    print(f"Processing complete in {time.time() - start_time:.2f} seconds.")
