import sqlite3
import uuid
import stanza
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import cProfile
import pstats

os.environ["OMP_NUM_THREADS"] = "1"

DB_PATH = '/data/fsg/.sumonlp/sentence_generation/LatestTrainingSet/vocabulary.db'
SENTENCE_PATH = '/data/fsg/.sumonlp/sentence_generation/LatestTrainingSet/combined-eng.txt'

# Initialize Stanza pipeline globally per worker
nlp = None

def init_worker():
    global nlp
    nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma', use_gpu=False, verbose=False, num_threads=1)

def get_word_type(word):
    if word.upos == 'NOUN':
        return 'noun'
    elif word.upos == 'VERB':
        return 'verb'
    else:
        return None

def process_sentence(line):
    line = line.strip()
    if not line:
        return []

    result = []
    doc = nlp(line)
    for sent in doc.sentences:
        for word in sent.words:
            word_type = get_word_type(word)
            if word_type:
                result.append((word.lemma, word_type, str(uuid.uuid4())))
    return result

def insert_dictionary(cursor, conn, all_words):
    print('Process of inserting the Dictionary values to DB started:')
    try:
        cursor.executemany('''
            INSERT OR IGNORE INTO Word (id, root, pos)
            VALUES (?, ?, ?)
        ''', all_words)
    except Exception as e:
        print(f"An error occurred at insert_dictionary: {e}")
    conn.commit()
    print('Process of inserting the Dictionary values to DB finished.')

def line_generator():
    with open(SENTENCE_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            yield line




if __name__ == "__main__":

    with cProfile.Profile() as profile:

      stanza.download('en')
      conn = sqlite3.connect(DB_PATH)
      cursor = conn.cursor()

      cursor.execute("""
      CREATE TABLE IF NOT EXISTS Word (
          id TEXT PRIMARY KEY,
          root TEXT NOT NULL,
          pos TEXT NOT NULL
      );
      """)
      conn.commit()

      cursor.execute("""
      CREATE INDEX IF NOT EXISTS idx_word_root ON Word (root);
      """)
      conn.commit()

      # Count total lines for tqdm
      with open(SENTENCE_PATH, 'r', encoding='utf-8') as f_in:
          total_lines = sum(1 for _ in f_in)

      print(f"Total sentences: {total_lines}")
      print('Sentences NLP process started:')

      # Multiprocessing pool
      all_pairs = set()
      with Pool(processes=16, initializer=init_worker) as pool:
          for result in tqdm(pool.imap_unordered(process_sentence, line_generator(), chunksize=500), total=total_lines, desc="Processing sentences"):
              for lemma, word_type, uuid_str in result:
                  all_pairs.add((lemma, word_type, uuid_str))

      print('Sentences NLP process finished.')

      # Prepare unique words for DB insert
      unique_dict = {}
      for lemma, word_type, uuid_str in all_pairs:
          if (lemma, word_type) not in unique_dict:
              unique_dict[(lemma, word_type)] = uuid_str

      # Convert to list for DB insert
      all_words = [(value, key[0], key[1]) for key, value in unique_dict.items()]

      insert_dictionary(cursor, conn, all_words)


    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TOTAL_TIME)
    results.print_stats()
