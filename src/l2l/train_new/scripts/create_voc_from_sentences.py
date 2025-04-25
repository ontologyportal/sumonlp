import sqlite3
import uuid
import stanza

DB_PATH = '/home/angelos.toutsios.gr/data/Train_T5_Model/data/500k_sentences_suffled/vocabulary.db'
SENTENCE_PATH = '/home/angelos.toutsios.gr/data/Train_T5_Model/data/500k_sentences_suffled/input_sentences_500k.txt'

nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma')

dictionary = {}

def insert_dictionary(cursor, conn):

  print('Process of inserting the Dictionary values to DB started:')

  for key, value in dictionary.items():
    try:
      cursor.execute('''
      INSERT INTO Word (id, root, pos)
      VALUES (?, ?, ?)
      ''', (value, key[0], key[1]))
    except Exception as e:
      print(f"An error occurred at insert_dictionary: {e}")

  conn.commit()

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


def process_sentence(sentence, conn, cursor):
    print('Sentences nlp process started:')
    doc = nlp(sentence)
    print('Sentences nlp process finished:')
    # Check NOUNS/VERBS in Vocabulary
    for sent in doc.sentences:
        for word in sent.words:
            word_type = get_word_type(word)
            if word_type:
                # Check if the root form exists in the Dictionary
                  if (word.lemma, word_type) not in dictionary:
                    dictionary[(word.lemma, word_type)] = str(uuid.uuid4())


def insert_dictionary():

  print('Process of inserting the Dictionary values to DB started:')

  for key, value in dictionary.items():
    try:
      cursor.execute('''
      INSERT INTO Word (id, root, pos)
      VALUES (?, ?, ?)
      ''', (value[0], key, value[1]))
    except Exception as e:
      print(f"An error occurred at insert_dictionary: {e}")

  conn.commit()


if __name__ == "__main__":
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

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Word (
        id TEXT PRIMARY KEY,  -- Use TEXT to store UUIDs
        root TEXT NOT NULL,
        pos TEXT NOT NULL);
    """)
    conn.commit()

    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_word_root ON Word (root);
    """)
    conn.commit()

    count = 0
    with open(SENTENCE_PATH, 'r', encoding='utf-8') as f_in:
      document = f_in.read()
      print('Sentences process started:')
      process_sentence(document, conn, cursor)
      # for line in f_in:
      #   count = count + 1
      #   stripped_line = line.strip()
      #   if stripped_line:
      #     process_sentence(stripped_line, conn, cursor)
      #     if count % 100 == 0:
      #       print('sentences processed:',count)

    insert_dictionary(cursor, conn)