import re
import sqlite3
import time
import os

# db_path = '/data/angelos.toutsios.gr/vocabulary.db'
DB_PATH = os.environ['VOCABULARY_HOME']+"/vocabulary.db"

# db_path = '/home/angelos.toutsios.gr/workspace/sumonlp/src/oov_handling/vocabulary_test.db'

def find_sumo_category(word_type):
  if word_type == 'DATE':
    return None
  elif word_type == 'EVENT':
    return 'SocialInteraction'
  elif word_type == 'FAC':
    return 'Facility'
  elif word_type == 'GPE':
    return 'GeopoliticalArea'
  elif word_type == 'LANGUAGE':
    return 'NaturalLanguage'
  elif word_type == 'LAW':
    return 'DeonticAttribute'
  elif word_type == 'LOC':
    return 'GeographicArea'
  elif word_type == 'MONEY':
    return 'CurrencyMeasure'
  elif word_type == 'NORP':
    return 'GroupOfPeople'
  elif word_type == 'ORG':
    return 'Organization'
  elif word_type == 'PERSON':
    return 'Human'
  elif word_type == 'PRODUCT':
    return 'Product'
  elif word_type == 'WORK_OF_ART':
    return 'ArtWork'
  else:
    return None


def format_word(word):

    word  = word.strip()

    # Step 1: Replace special characters (except commas and periods) with underscores
    word = re.sub(r'[^a-zA-Z0-9,.!?\s]', '_', word)

    # Step 2: Remove commas and periods
    word = word.replace(',', '').replace('.', '').replace('!','').replace('?','')

    # Step 3: Add prefix 'num_' if the word starts with a digit

    if word[0].isdigit():
        word = 'num_' + word

    # Step 4: Replace any remaining whitespaces with underscores
    word = word.replace(' ', '_')

    return word


def get_word_from_db(word_id, conn, cursor):

  try:
    # Query to get the word based on ID
    cursor.execute("SELECT word, type FROM UnknownWords WHERE id = ?", (word_id,))
    result = cursor.fetchone()

    # If the word is found, mark it as used and return it
    if result:
        word = result[0]
        word_type = result[1]
        word = format_word(word)
        cursor.execute("UPDATE UnknownWords SET Used = 1, formatted_word = ? WHERE id = ?", (word, word_id))
        conn.commit()
        return (word, word_type)
    else:
        return None
  except sqlite3.Error as e:
        # Handle any SQLite errors
        print(f"Database error: {e}")
        return None
  except Exception as e:
        # Handle unexpected errors
        print(f"Unexpected error: {e}")
        return None

def replace_unk_words(input_file, output_file, conn, cursor):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    words_replaced = []

    # Regex pattern to find all <UNK_word_type_ID> words
    # pattern = r'<UNK_(\w+)_([\d]+)>'
    pattern = r'UNK_(\w+)_([\d]+)'

    # Replace all matches in the content
    def replacement(match):
        word_id = match.group(2)
        (replacement_word, word_type) = get_word_from_db(word_id, conn, cursor)
        if replacement_word is not None and (replacement_word, word_type) not in words_replaced:
          words_replaced.append((replacement_word, word_type))
        return replacement_word if replacement_word else match.group(0)

    updated_content = re.sub(pattern, replacement, content)

    # Write the updated content to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        for replacement_word, word_type in words_replaced:
          sumo_term = find_sumo_category(word_type)
          if sumo_term == 'Human':
            file.write(f'( and ( instance {replacement_word} {sumo_term} ) (names \"{replacement_word}\" {sumo_term} ) )\n')
          elif sumo_term is not None:
            file.write(f'(instance {replacement_word} {sumo_term})\n')
        file.write(updated_content)

if __name__ == "__main__":

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    start_time = time.time()

    input_filename = 'input_post_oov.txt'
    output_filename = 'output_post_oov.txt'
    replace_unk_words(input_filename, output_filename, conn, cursor)
    conn.close()

    print(f"Processing complete in {time.time() - start_time:.2f} seconds.")
