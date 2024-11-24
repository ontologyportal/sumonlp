import sqlite3
import uuid
import os


#DB connection
conn = sqlite3.connect('vocabulary.db')

dictionary = {}

def split_sentences(filename):

    sentences = []
    current_sentence = []

    # Open and read the file
    with open(filename, 'r', encoding='latin1') as file:
        for line in file:
            # Split the line into parts
            parts = line.strip().split('\t')

            # We have 2 kind of files with different number of columns, 3 columns and 5 columns
            if len(parts) != 3 and len(parts) != 5:
                continue
            elif len(parts) == 3:
                word_info = [parts[0], parts[1], parts[2]] # [word, root, pos]
                current_sentence.append(word_info)
            elif len(parts) == 5:
                # Extract word, root, and POS tag
                word_info = [parts[2], parts[3], parts[4]]
                current_sentence.append(word_info)

            # Check if this word ends the sentence
            if word_info[0] in {'.', '!', '?', '#', '<p>', '...', '....'} :
                sentences.append(current_sentence)
                current_sentence = []

    # Add the last sentence if it doesn't end with punctuation
    if current_sentence:
        sentences.append(current_sentence)
    return sentences


def create_relations(sentences):

    for sentence in sentences:
        verbs = []
        nouns = []

        for word_info in sentence:
            word, root, pos = word_info
            if root == '':
                continue
            # if pos.lower().startswith('vb') or pos.lower().startswith('vv'):
            #     pos = 'verb'
            #     if (root, pos) not in dictionary:
            #         dictionary[(root, pos)] = str(uuid.uuid4())

            # elif pos.lower().startswith('nn'):
            #     pos = 'noun'
            #     if (root, pos) not in dictionary:
            #         dictionary[(root, pos)] = str(uuid.uuid4())

            elif pos.lower().startswith('np'): # names
                pos = 'noun-phrase'
                if (root, pos) not in dictionary:
                    dictionary[(root, pos)] = str(uuid.uuid4())

def insert_dictionary():

  print('Process of inserting the Dictionary values to DB started:')

  for key, value in dictionary.items():
    try:
      cursor.execute('''
      INSERT INTO Word (id, root, pos)
      VALUES (?, ?, ?)
      ''', (value, key[0], key[1]))
    except Exception as e:
      print(f"An error occurred at insert_dictionary: {e}: the word {key[0]}, {key[1], {value}}")
  conn.commit()
  print('Insert in dictionary completed')


# Using the special variable
# __name__
if __name__=="__main__":

    cursor = conn.cursor()

    for root, dirs, files in os.walk('/home/angelos.toutsios.gr/data/Thesis_dev/COCA_statistics/COCA'):
        counter = 0
        for file in files:
            if file.endswith('.txt'):
                filename = os.path.join(root, file)
                counter += 1
                print(f'Processing file {counter}/{len(files)} | name: {filename}')
                sentences = split_sentences(filename)
                create_relations(sentences)

    insert_dictionary()





