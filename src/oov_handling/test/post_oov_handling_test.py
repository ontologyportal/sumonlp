import unittest
import sqlite3
import tempfile
import os
import sys
from unittest.mock import patch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock the environment variable before importing the module
with patch.dict(os.environ, {"SUMO_NLP_HOME": "/mocked/path"}):
    from post_oov_handling import format_word, replace_unk_words, clear_unknown_words_from_db


class TestMainCode(unittest.TestCase):

    def setUp(self):
        """Set up an in-memory database and temporary files for testing."""
        # In-memory SQLite database
        self.conn = sqlite3.connect(":memory:")
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS UnknownWords (
            id INTEGER,
            sentence_id INTEGER,
            word TEXT,
            type TEXT DEFAULT '',
            PRIMARY KEY (id, sentence_id, type)
        )
        """)
        self.conn.commit()

        # Insert test data
        self.test_data = [
            (1,1, "Paris", "GPE"),
            (2,1, "John", "PERSON"),
            (3,1, "Google", "ORG"),
            (4,1, "123Main", "LOC"),
        ]
        self.cursor.executemany("INSERT INTO UnknownWords (id, sentence_id, word, type) VALUES (?, ?, ?, ?)", self.test_data)
        self.conn.commit()

        # Create temporary files for input and output
        self.input_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='utf-8')
        self.output_file = tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='utf-8')

        # Add test content to the input file

        self.input_file.write("SentenceId:1\n")
        self.input_file.write("UNK_GPE_1 UNK_PERSON_2 UNK_ORG_3 UNK_LOC_4")
        self.input_file.seek(0)

    def tearDown(self):
        """Clean up temporary files and close the database connection."""
        self.input_file.close()
        self.output_file.close()
        os.unlink(self.input_file.name)
        os.unlink(self.output_file.name)
        self.conn.close()

    def test_replace_unk_words(self):
        """Test the replace_unk_words function."""
        replace_unk_words(self.input_file.name, self.output_file.name, self.conn, self.cursor)

        # Check output file content
        with open(self.output_file.name, 'r', encoding='utf-8') as output:
            content = output.read()

        # Expected SUMO mappings and replacement content
        expected_lines = [
            "( instance Paris GeopoliticalArea )\n",
            "( and ( instance John Human ) (names \"John\" John ) )\n"
            "( instance Google Organization )\n",
            "( instance num_123Main GeographicArea )\n",
        ]
        for line in expected_lines:
            self.assertIn(line, content)

        self.assertIn("Paris John Google num_123Main", content)

    def test_format_word(self):
        """Test the format_word function with various inputs."""
        test_cases = [
            ("Hello", "Hello"),  # No changes needed
            ("Hello, World!", "Hello_World"),  # Remove commas and replace special characters
            ("123MainSt", "num_123MainSt"),  # Prefix 'num_' if starts with a digit
            ("  spaces  ", "spaces"),  # Replace spaces with underscores and strip whitespace
            ("Special@Char$", "Special_Char_"),  # Replace special characters with underscores
            ("12.34%", "num_1234_"),  # Combine multiple transformations
            # ("", ""),  # Empty string
            ("A_B.C,D", "A_BCD"),  # Remove commas and periods, keep letters/numbers
            ("123test", "num_123test"),  # Multiple transformations in sequence
        ]

        for input_word, expected_output in test_cases:
            with self.subTest(input=input_word, expected=expected_output):
                self.assertEqual(format_word(input_word), expected_output)

    def test_clear_unknown_words_from_db(self):
        """Test the clear_unknown_words_from_db function."""
        # Check initial count
        self.cursor.execute("SELECT COUNT(*) FROM UnknownWords")
        initial_count = self.cursor.fetchone()[0]
        self.assertEqual(initial_count, 4)  # Ensure test data is inserted

        # Call the function to clear the table
        clear_unknown_words_from_db(self.conn, self.cursor)

        # Verify the table is empty
        self.cursor.execute("SELECT COUNT(*) FROM UnknownWords")
        final_count = self.cursor.fetchone()[0]
        self.assertEqual(final_count, 0)


if __name__ == "__main__":
    unittest.main()
