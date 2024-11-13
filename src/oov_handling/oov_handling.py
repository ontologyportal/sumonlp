import sqlite3
import stanza
import os
import time

# Initialize Stanza pipeline for English
stanza.download('en', processors='tokenize,pos,lemma,ner', verbose=False)  # Download model if not already done
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner', verbose=False)

# Database configuration
# DB_PATH = '/data/angelos.toutsios.gr/vocabulary_test.db'
DB_PATH = '/home/angelos.toutsios.gr/workspace/sumonlp/src/oov_handling/vocabulary_test.db'


def get_word_type(word):
    """Determine if the word is a noun or verb using Stanza."""
    if word.upos == 'NOUN':
        return 'noun'
    # elif word.upos == 'PROPN':
    #   return 'noun-phrase'
    elif word.upos == 'VERB':
        return 'verb'
    else:
        return None

def check_word_in_dictionary(root, word_type, cursor):
    """Check if the word exists in the Dictionary table."""
    cursor.execute("SELECT id FROM Word WHERE root = ? AND pos = ?", (root.lower(), word_type))
    # cursor.execute("SELECT id FROM Dictionary WHERE root = ?", (root.lower()))
    return cursor.fetchone()

def add_unknown_word(word, word_type, conn, cursor):
    """Check if an unknown word already exists in the UnknownWords table with 'used' set to 0, or insert it if not."""
    try:
        # trim word
        word = word.strip()
        word = word.lower()
        cursor.execute("SELECT * FROM UnknownWords WHERE word = ? AND used = 0 AND type = ?", (word, word_type))
        result = cursor.fetchone()
        if result:
            return result[0] # Return the ID if the word exists
        else:
            cursor.execute("INSERT INTO UnknownWords (word, type) VALUES (?,?)", (word, word_type))
            conn.commit()
            return cursor.lastrowid # Return the ID of the newly inserted word
    except sqlite3.IntegrityError as e:
        print(f"Error: {e}")
        return None


def process_sentence(sentence, conn, cursor):
    """Process a single sentence, replacing unknown nouns and verbs with tags."""

    doc = nlp(sentence)
    processed_tokens = []

    # Check NOUNS/VERBS in Vocabulary
    for sent in doc.sentences:
        for word in sent.words:
            word_type = get_word_type(word)
            if word_type:
                # Check if the root form exists in the Dictionary
                dictionary_entry = check_word_in_dictionary(word.lemma, word_type, cursor)
                if dictionary_entry:
                    # Known word, keep the original token text
                    processed_tokens.append(word.text)
                else:
                    # Unknown word, replace with <UNK_type_id>
                    unk_id = add_unknown_word(word.text, word_type, conn, cursor)
                    if unk_id != None:
                        processed_tokens.append(f"<UNK_{word_type}_{unk_id}>")
                    else:
                        # In case of any issue, keep the original word
                        processed_tokens.append(word.text)
            else:
                # Non-noun/verb word, keep as-is
                processed_tokens.append(word.text)

    # Join processed tokens to form the sentence with appropriate spacing
    new_sentence = ' '.join(processed_tokens)

    doc = nlp(new_sentence)

    # NER process  {ent.text} {ent.type}
    for ent in doc.ents:

      # Save each ent and type in DB
      unk_id = add_unknown_word(ent.text, ent.type, conn, cursor)

      # Replace in the document each ent with the appropiate <tag>
      tag = f'<UNK_{ent.type}_{unk_id}>'
      new_sentence = new_sentence.replace(ent.text, tag)

    return new_sentence




def process_file(input_file, output_file, conn, cursor):
    """Process the input file and write the processed text to the output file."""
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            stripped_line = line.strip()
            if stripped_line:
                processed_line = process_sentence(stripped_line, conn, cursor)
                outfile.write(processed_line + '\n')
            else:
                outfile.write('\n')  # Preserve empty lines



if __name__ == "__main__":
    start_time = time.time()
    input_filename = 'input_oov.txt'    # Input file containing sentences
    output_filename = 'output_oov.txt'  # Output file to save processed sentences


    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure the Dictionary and UnknownWords tables exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS UnknownWords (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        word TEXT UNIQUE,
        formatted_word TEXT DEFAULT '',
        type TEXT DEFAULT '',
        used INTEGER DEFAULT 0 CHECK (used IN (0, 1))
    )
    """)
    conn.commit()

    process_file(input_filename, output_filename, conn, cursor)

    # Close the database connection after processing
    conn.close()
    print(f"Processing complete in {time.time() - start_time:.2f} seconds.")
