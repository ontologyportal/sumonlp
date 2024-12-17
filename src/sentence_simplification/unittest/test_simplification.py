import unittest
import os, sys
from unittest.mock import patch, mock_open
import ollama
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from simplification import call_ollama, simplify_file, remove_pronouns, simplify_sentence_llm, split_sentences


class TestSimplification(unittest.TestCase):
    def setUp(self):
        self.model_type = 'llama3.1'

    # Mock cases

    # call ollama tests
    @patch("simplification.ollama.chat")  # Mock the ollama.chat function
    def test_call_ollama_valid_response(self, mock_ollama):
        mock_ollama.return_value = {"message": {"content": "Simplified sentence."}}
        response = call_ollama("Simplify this sentence.", self.model_type)
        self.assertEqual(response, "Simplified sentence.")
    
    @patch("simplification.ollama.chat")
    def test_call_ollama_handles_error(self, mock_ollama):
        mock_ollama.side_effect = Exception("API error")
        response = call_ollama("Simplify this sentence.", self.model_type)
        self.assertIsNone(response)  

    # remove pronouns
    @patch("simplification.call_ollama")  # Mock call_ollama since it's used inside remove_pronouns
    def test_remove_pronouns(self, mock_call_ollama):
        mock_call_ollama.return_value = "The dog chased the ball and jumped on the ball."
        sentence = "The dog chased the ball and jumped on it."
        result = remove_pronouns(sentence, self.model_type)
        self.assertEqual(result, "The dog chased the ball and jumped on the ball.")

    # split_sentences function
    @patch("simplification.stanza.Pipeline")  # Mock stanza pipeline
    def test_split_sentences(self, mock_pipeline):
        mock_pipeline.return_value = mock_pipeline
        mock_pipeline.return_value(text="Sentence one. Sentence two.")
        mock_pipeline.return_value.sentences = [mock_pipeline]
        mock_pipeline.sentences = [
            type("Sentence", (), {"text": "Sentence one."}),
            type("Sentence", (), {"text": "Sentence two."}),
        ]
        result = split_sentences("Sentence one. Sentence two.")
        self.assertEqual(result, ["Sentence one.", "Sentence two."])

    # Integration tests using actual ollama

    def test_simplify_sentence_llm(self):
        sentence = "A huge machine that levels the tracks leading into Grand Central Terminal snagged a 600-volt electrical cable in a tunnel beneath Park Avenue yesterday, sending up a shower of sparks and flames that injured 11 railroad workers and 2 firefighters."
        result = simplify_sentence_llm(sentence, self.model_type)
        self.assertIsInstance(result, str)

    def test_remove_pronouns(self):
        sentence = "The dog chased the ball and jumped on it."
        result = remove_pronouns(sentence, self.model_type)
        self.assertIsInstance(result, str)
        self.assertEqual(result, "The dog chased the ball and jumped on the ball.")

    def test_split_sentences(self):
        sentence = "Sentence one. Sentence two."
        result = split_sentences(sentence)
        self.assertIsInstance(result, list)
        self.assertEqual(result, ["Sentence one.", "Sentence two."])

        sentence = "Dr. John and Mary D. Anderson went to the park. They had a picnic. Mrs. Anderson brought her dog."
        result = split_sentences(sentence)
        self.assertIsInstance(result, list)
        self.assertEqual(result, ["Dr. John and Mary D. Anderson went to the park.", "They had a picnic.", "Mrs. Anderson brought her dog."])

if __name__ == "__main__":
    unittest.main()


