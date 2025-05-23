import sqlite3
import stanza
import os
import time
import warnings
from tqdm import tqdm
import cProfile

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


# Initialize Stanza pipeline for English
stanza.download('en', processors='tokenize,pos,lemma,ner', verbose=False)  # Download model if not already done
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner', verbose=False)

# Database configuration
DB_PATH = os.environ['SUMO_NLP_HOME']+"/vocabulary.db"

sentence_id = 1 # Sentence Counter

def get_word_type(word):
    """Determine if the word is a noun or verb using Stanza."""
    if word.upos == 'NOUN':
        return 'noun'
    # PROPN is handling from NER
    # elif word.upos == 'PROPN':
    #   return 'noun-phrase'
    elif word.upos == 'VERB':
        return 'verb'
    else:
        return None

def check_word_in_dictionary(root, word_type, cursor):
    """Check if the word exists in the Dictionary table."""
    cursor.execute("SELECT id FROM Word WHERE root = ? AND pos = ?", (root.lower(), word_type))
    return cursor.fetchone()


def get_max_id_from_db(cursor, sent_id, word_type):
  try:
    cursor.execute("SELECT max(id) FROM UnknownWords WHERE sentence_id = ? AND type = ?", (sent_id, word_type))
    result = cursor.fetchone()
    if result:
      return result[0]
    else:
      return 0
  except sqlite3.IntegrityError as e:
      print(f"Error in def: get_max_id_from_db : {e}")
      print(f"Errors caused from: sentId: {sent_id}, WordType: {word_type}")
      print(f"result:{result}")
      return None


def add_unknown_word(word, word_type, conn, cursor, sent_id):
    """Check if an unknown word already exists in the UnknownWords table, or insert it if not."""
    try:
        # trim word
        word = word.strip()
        word = word.lower()
        # print("word is: " + word)
        cursor.execute("SELECT * FROM UnknownWords WHERE word = ? AND type = ? and sentence_id = ?", (word, word_type, sent_id))
        result = cursor.fetchone()
        if result:
            return (result[0],'exist') # Return the ID if the word exists
        else:
            max_id = get_max_id_from_db(cursor, sent_id, word_type)
            if max_id:
              word_id = max_id + 1
            else:
              word_id = 1
            cursor.execute("INSERT INTO UnknownWords (id, sentence_id, word, type) VALUES (?,?,?,?)", (word_id, sent_id, word, word_type))
            conn.commit()
            return (word_id,'new') # Return the ID of the newly inserted word
    except sqlite3.IntegrityError as e:
        print(f"Error: {e}")
        print(f"Word that caused the error: {word}")
        print(f"result:{result}")
        return None


def process_sentence(sentence, conn, cursor):
    """Process a single sentence, replacing unknown nouns and verbs with tags."""

    doc = nlp(sentence)
    sentences = []
    global sentence_id

    # Check NOUNS/VERBS in Vocabulary
    for sent in doc.sentences:
      processed_tokens = []
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

                  id_exist = add_unknown_word(word.text, word_type, conn, cursor, sentence_id)
                  if id_exist != None:
                      processed_tokens.append(f"UNK_{word_type}_{id_exist[0]}")
                  else:
                      # In case of any issue, keep the original word
                      processed_tokens.append(word.text)
          else:
              # Non-noun/verb word, keep as-is
              processed_tokens.append(word.text)

      # Join processed tokens to form the sentence with appropriate spacing
      new_sentence = ' '.join(processed_tokens)
      # nlp again the sentence
      doc = nlp(new_sentence)
      # NER process  {ent.text} {ent.type}

      ner_entities_excluded = ["DATE","CARDINAL","PERCENT", "ORDINAL", "QUANTITY","TIME"]

      for ent in doc.ents:
        # Don't try to NER the Unknown Words from previous step
        if ent.type not in ner_entities_excluded and not ent.text.startswith("UNK_"):
          # Save each ent and type in DB
          id_exist = add_unknown_word(ent.text, ent.type, conn, cursor, sentence_id)

          # Replace in the document each ent with the appropiate <tag>
          if id_exist != None:
            tag = f'UNK_{ent.type}_{id_exist[0]}'
            new_sentence = new_sentence.replace(ent.text, tag)


      sentences.append(f"SentenceId:{sentence_id}")
      sentences.append(new_sentence)
      sentence_id += 1

    return "\n".join(sentences)


def process_file(input_file, output_file, conn, cursor):
    """Process the input file and write the processed text to the output file."""
    print(f"Processing file: {input_file}")
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' does not exist.")
        return

    with open(input_file, 'r', encoding='utf-8') as infile:
        total_lines = sum(1 for _ in infile)
        infile.seek(0)

        with open(output_file, 'w', encoding='utf-8') as outfile:
            for line in tqdm(infile, total=total_lines, desc="Processing lines"):
                stripped_line = line.strip()
                if stripped_line:
                    processed_line = process_sentence(stripped_line, conn, cursor)
                    outfile.write(processed_line + '\n')

def clear_unknown_words_from_db(conn, cursor):
  try:
    # Query to get the word based on ID
    cursor.execute("DELETE FROM UnknownWords")
    cursor.execute("SELECT COUNT(*) FROM UnknownWords")
    result = cursor.fetchone()
    # If the word is found, mark it as used and return it
    if result[0] == 0:
      print(f"Database cleaned from Unknown Words")
      conn.commit()
    else:
      print(f"ERROR Cleaning Database from Unknown Words. number of words found: {result[0]}")
  except sqlite3.Error as e:
        # Handle any SQLite errors
        print(f"Database error: {e}")
  except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error: {e}")


def main():
    start_time = time.time()
    input_filename = 'input_oov.txt'    # Input file containing sentences
    output_filename = 'output_oov.txt'  # Output file to save processed sentences


    # Connect to the SQLite database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure the Dictionary and UnknownWords tables exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS UnknownWords (
        id INTEGER,
        sentence_id INTEGER,
        word TEXT,
        type TEXT DEFAULT '',
        PRIMARY KEY (id, sentence_id, type)
    )
    """)
    conn.commit()

    clear_unknown_words_from_db(conn, cursor)
    process_file(input_filename, output_filename, conn, cursor)

    # Close the database connection after processing
    conn.close()
    print(f"Processing complete in {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()