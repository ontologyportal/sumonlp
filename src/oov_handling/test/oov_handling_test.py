import unittest
import sqlite3
import stanza
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from oov_handling import get_word_type, check_word_in_dictionary, add_unknown_word, process_sentence

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
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT UNIQUE,
            formatted_word TEXT DEFAULT '',
            type TEXT DEFAULT '',
            used INTEGER DEFAULT 0 CHECK (used IN (0, 1))
        )""")
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

    def test_add_unknown_word(self):
        """Test add_unknown_word function."""
        word = "unknownword"
        word_type = "noun"

        # Check if the word was added
        unk_id = add_unknown_word(word, word_type, self.conn, self.cursor)
        self.assertIsNotNone(unk_id)

        # Try adding the same word again, should return the same ID
        existing_id = add_unknown_word(word, word_type, self.conn, self.cursor)
        self.assertEqual(unk_id, existing_id)

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
          ("I like to run and jump", "I like to run and <UNK_verb_1>"),  # 'jump' is unknown
          ("They play soccer", "They play <UNK_noun_2>"),  # All words are known
          ("We hike and swim", "We <UNK_verb_3> and <UNK_verb_4>"),  # 'hike' and 'swim' are unknown
          # 'dog' is known, 'park' is unknown (noun), and 'John' (NER) should be tagged
          ("John and his dog went to the park", " <UNK_PERSON_6> and his dog went to the <UNK_noun_5>")
      ]

      for sentence, expected_output in test_cases:
          # Process the sentence
          processed_sentence = process_sentence(sentence, self.conn, self.cursor)

          # Assert the processed sentence matches the expected output
          self.assertEqual(processed_sentence.strip(), expected_output.strip(), f"Failed for sentence: '{sentence}'")


if __name__ == '__main__':
    unittest.main()