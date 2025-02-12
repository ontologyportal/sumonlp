import unittest
from mock import MagicMock, patch
import sys
import re
import ollama
from transformers import pipeline
import torch
from nltk.translate.bleu_score import sentence_bleu
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metaphor_detect_pipeline import MetaphorDetector
from metaphor_trans import MetaphorTranslator


class TestDetection(unittest.TestCase):

    def setUp(self):
        self.detector = MetaphorDetector()

    def test_tokenization(self):
        sentence = "Time flies like an arrow."
        label_list = self.detector.detect_metaphor(sentence)[1] # Simulating tokenization
        self.assertEqual(len(label_list), 6, "Token count should be 6.")

    def test_metaphor_classification(self):
        self.detector.detect_metaphor = MagicMock(side_effect=[1, 0, 0, 0])

        self.assertEqual(self.detector.detect_metaphor("He’s carrying the weight of the world."), 1, "Metaphorical sentence should be classified as 1.")
        self.assertEqual(self.detector.detect_metaphor("He has many responsibilities."), 0, "Literal sentence should be classified as 0.")
        self.assertEqual(self.detector.detect_metaphor("His plate is full."), 0, "Ambiguous metaphor should be classified as 0.")
        self.assertEqual(self.detector.detect_metaphor("!!!!"), 0, "Nonsensical sentence should be classified as 0.")

class TestTranslation(unittest.TestCase):

    def setUp(self):
        model_type = 'llama3.2' # experiment with different models here
        self.translator = MetaphorTranslator(model_type)

    @patch('metaphor_trans.MetaphorTranslator.call_ollama')
    def test_mock_ollama_error_handling(self, mock_call_ollama):
        mock_call_ollama.side_effect = Exception("API error")
        result = self.translator.translate_metaphor("He’s carrying the weight of the world.")
        self.assertIsNone(result, "Translation should be None if Ollama encounters an error.")

    @patch('metaphor_trans.MetaphorTranslator.translate_metaphor')
    def test_mock_ollama_translation(self, mock_translate):
        mock_translate.side_effect = ["He has many responsibilities."]

        result = self.translator.translate_metaphor("1 He’s carrying the weight of the world.")
        self.assertEqual(result, "He has many responsibilities.", "Translation should match Ollama output.")

    @patch('metaphor_trans.MetaphorTranslator.translate_metaphor')
    def test_no_translation_for_literal_sentence(self, mock_translate):
        mock_translate.side_effect = ["He has many responsibilities."]  # Will not be used

        input_sentence = "0 He has many responsibilities."
        result = self.translator.translate_metaphor(input_sentence)
        self.assertEqual(result, "He has many responsibilities.", "Literal sentences should not be translated.")

    def test_actual_ollama_bleu_score(self):
        # Assuming we have a working method for `translate_metaphor`
        original_sentence = "He’s carrying the weight of the world."
        translated_sentence = self.translator.translate_metaphor(original_sentence)

        reference_translation = ["He has many responsibilities.".split()]
        candidate_translation = translated_sentence.split()

        bleu_score = sentence_bleu(reference_translation, candidate_translation)

        self.assertLess(bleu_score, 0.5, "BLEU score should indicate adequate translation quality.")

if __name__ == '__main__':
    unittest.main()
