import unittest
import sqlite3
import stanza
import os
import sys
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

with patch.dict(os.environ, {"SUMO_NLP_HOME": "/mocked/path"}):
  from oov_handling import get_word_type, check_word_in_dictionary, add_unknown_word, process_sentence, get_max_id_from_db

# Path to a temporary test database
TEST_DB_PATH = ':memory:'

class VocabularyTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a temporary SQLite database and Stanza pipeline."""
        stanza.download('en', processors='tokenize,pos,lemma,ner', verbose=False)
        cls.nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,ner', verbose=False)

        # Connect to test database
        cls.conn = sqlite3.connect(TEST_DB_PATH)
        cls.cursor = cls.conn.cursor()

        # Create tables
        cls.cursor.execute("""
        CREATE TABLE IF NOT EXISTS Word (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            root TEXT,
            pos TEXT
        )""")
        cls.cursor.execute("""
          CREATE TABLE IF NOT EXISTS UnknownWords (
              id INTEGER,
              sentence_id INTEGER,
              word TEXT,
              type TEXT DEFAULT '',
              PRIMARY KEY (id, sentence_id, type)
          )
          """)
        cls.conn.commit()

    @classmethod
    def tearDownClass(cls):
        """Close the database connection and delete the test database file."""
        cls.conn.close()
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)

    def test_get_word_type(self):
        """Test get_word_type function."""
        # Assume we have a sample Stanza word with a verb POS
        doc = self.nlp("jump")
        word = doc.sentences[0].words[0]
        self.assertEqual(get_word_type(word), 'verb')

    def test_check_word_in_dictionary(self):
        """Test check_word_in_dictionary function."""
        # Insert a sample word into Word table
        self.cursor.execute("INSERT INTO Word (root, pos) VALUES (?, ?)", ("run", "verb"))
        self.conn.commit()

        # Check if the word exists
        result = check_word_in_dictionary("run", "verb", self.cursor)
        self.assertIsNotNone(result)

    def test_get_max_id_no_result(self):

        # Define test input
        sent_id = 2
        word_type = "verb"

        # Test the method
        result = get_max_id_from_db(self.cursor, sent_id, word_type)

        # Assert the correct fallback value is returned
        self.assertIsNone(result)


    def test_get_max_id_from_db(self):

        # Define test input
        sent_id = 1
        word_type = "noun"
        word_id = 12
        word = 'test'

        self.cursor.execute("INSERT INTO UnknownWords (id, sentence_id, word, type) VALUES (?,?,?,?)", (word_id, sent_id, word, word_type))

        # Test the method
        result = get_max_id_from_db(self.cursor, sent_id, word_type)

        self.assertEqual(result, 12)


    def test_add_unknown_word(self):
        """Test add_unknown_word function."""
        word = "unknownword"
        word_type = "noun"

        sent_id = 1
        # Check if the word was added
        unk_id = add_unknown_word(word, word_type, self.conn, self.cursor, sent_id)
        self.assertEqual(unk_id[1],'new')

        # Try adding the same word again, should return the same ID
        existing_id = add_unknown_word(word, word_type, self.conn, self.cursor, sent_id)
        self.assertEqual(existing_id[0], 1)


    def test_process_multiple_sentences(self):
      # """Test process_sentence function with multiple sentences."""
      # Set up known words in the Word table
      self.cursor.execute("DELETE FROM Word")  # Delete from Word table (or other relevant tables)
      self.cursor.execute("DELETE FROM UnknownWords")  # Delete from UnknownWords table (if necessary)

      self.cursor.execute("DELETE FROM sqlite_sequence WHERE name='UnknownWords'");

      self.cursor.execute("INSERT INTO Word (root, pos) VALUES (?, ?)", ("play", "verb"))
      self.cursor.execute("INSERT INTO Word (root, pos) VALUES (?, ?)", ("run", "verb"))
      self.cursor.execute("INSERT INTO Word (root, pos) VALUES (?, ?)", ("dog", "noun"))
      self.cursor.execute("INSERT INTO Word (root, pos) VALUES (?, ?)", ("go", "verb"))
      self.cursor.execute("INSERT INTO Word (root, pos) VALUES (?, ?)", ("like", "verb"))
      self.conn.commit()

      # Define multiple test cases with input sentences and expected outputs
      test_cases = [
          ("I like to run and jump", "SentenceId:1\nI like to run and UNK_verb_1"),  # 'jump' is unknown
          ("They play soccer", "SentenceId:2\nThey play UNK_noun_1"),  # All words are known
          ("We hike and swim", "SentenceId:3\nWe UNK_verb_1 and UNK_verb_2"),  # 'hike' and 'swim' are unknown
          # 'dog' is known, 'park' is unknown (noun), and 'John' (NER) should be tagged
          ("John and his dog went to the park", "SentenceId:4\nUNK_PERSON_1 and his dog went to the UNK_noun_1")
      ]

      for sentence, expected_output in test_cases:
          # Process the sentence
          processed_sentence = process_sentence(sentence, self.conn, self.cursor)

          # Assert the processed sentence matches the expected output
          self.assertEqual(processed_sentence.strip(), expected_output.strip(), f"Failed for sentence: '{sentence}'")


if __name__ == '__main__':
    unittest.main()
