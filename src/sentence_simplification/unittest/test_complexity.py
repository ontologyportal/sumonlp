import unittest
import os, sys
from unittest.mock import patch, mock_open
import ollama
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from complexity import *


class TestSimplification(unittest.TestCase):
    def setUp(self):
        self.model_type = 'llama3.1'

    def test_get_sentence_length(self):
        sentence = "The dog chased the ball and jumped on the ball."
        result = get_sentence_length(sentence)
        self.assertEqual(result, 10)

        sentence = "The     dog  chased   the        ball."
        result = get_sentence_length(sentence)
        self.assertEqual(result, 5)

    def test_check_pronouns_ollama(self):
        sentence = "The dog chased the ball and jumped on the ball."
        result = check_pronouns_ollama(sentence, self.model_type)
        print(sentence, result)  
        self.assertEqual(result, False)

        sentence = "The dog chased the ball and jumped on it."
        result = check_pronouns_ollama(sentence, self.model_type)
        print(sentence, result)
        self.assertEqual(result, True)

    def test_check_complexity_ollama(self):
        sentence = "The dog chased the ball and jumped on the ball."
        result = check_complexity_ollama(sentence, self.model_type)
        self.assertEqual(result, False)

        sentence = "The plaintiff's assertion of interference hinges upon demonstrating malfeasance wherein the defendant knowingly induced contractual breach through duplicitous means."
        result = check_complexity_ollama(sentence, self.model_type)
        self.assertEqual(result, True)

    def test_check_complexity_hueristic(self):
        sentence = "This is going to be a sentence over 20 words long, so it should be considered complex when passed through the hueristic function that was written."
        result = check_complexity_hueristic(sentence, 20)
        self.assertEqual(result, True)

if __name__ == "__main__":
    unittest.main()




