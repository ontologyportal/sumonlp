import re
import sqlite3
import time

db_path = '/data/angelos.toutsios.gr/vocabulary.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

def get_word_from_db(word_id):
    # Connect to the database

    # Query to get the word based on ID
    cursor.execute("SELECT word FROM UnknownWords WHERE id = ?", (word_id,))
    result = cursor.fetchone()

    # If the word is found, mark it as used and return it
    if result:
        word = result[0]
        cursor.execute("UPDATE UnknownWords SET Used = 1 WHERE id = ?", (word_id,))
        conn.commit()
        return word
    else:
        return None

def replace_unk_words(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regex pattern to find all <UNK_word_type_ID> words
    pattern = r'<UNK_(\w+)_([\d]+)>'

    # Replace all matches in the content
    def replacement(match):
        word_id = match.group(2)
        replacement_word = get_word_from_db(word_id)
        return replacement_word if replacement_word else match.group(0)

    updated_content = re.sub(pattern, replacement, content)

    # Write the updated content to the output file
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(updated_content)

if __name__ == "__main__":

    start_time = time.time()

    input_filename = 'input_post_oov.txt'
    output_filename = 'output_post_oov.txt'
    replace_unk_words(input_filename, output_filename)
    conn.close()

    print(f"Processing complete in {time.time() - start_time:.2f} seconds.")
