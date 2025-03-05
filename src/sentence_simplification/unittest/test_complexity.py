import unittest
import os, sys
from unittest.mock import patch, mock_open
import ollama
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from complexity import *


class TestSimplification(unittest.TestCase):
    def setUp(self):
        self.model_type = 'llama3.1:8b-instruct-q8_0'

    def test_get_sentence_length(self):
        sentence = "The dog chased the ball and jumped on the ball."
        result = get_sentence_length(sentence)
        self.assertEqual(result, 47)

        sentence = "The     dog  chased   the        ball."
        result = get_sentence_length(sentence)
        self.assertEqual(result, 5)

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

    def test_check_pronouns(self):
        ''' Test check_pronouns function '''
        model = 'llama3.1:8b-instruct-q8_0'

        test_cases = [
            ("The dog chased the ball and jumped on the ball.", False),
            ("John kicked the ball to Bill.", False),
            ("The dog chased the ball and jumped on it.", True),
            ("The government and fossil fuel industry must provide compensation for climate change-related damage.", False),
            ("The FIS has launched a Sustainability Guide for Ski Resorts. It helps resorts improve sustainability.", True),
            ("Extreme weather events result from global warming. They have become more frequent.", True),
            ("Mount Logan is in Canada. It is Canada's highest mountain.", True),
            ("The Intergovernmental Panel on Climate Change states that human activity influences the climate system.", False),
            ("The ice core samples were from pole to pole. They included those from Mount Logan.", True),
            ("Jane walks to the store. She buys groceries.", True),
            ("The weather is normally good. It rarely changes suddenly.", True),
            ("Adaptation measures can be implemented to address global warming. They are practical actions.", True),
            ("Vector-borne infectious diseases are caused by pathogens. The pathogens are transmitted by arthropods.", False),
            ("Polar ice reflects sunlight. It sends sunlight into space.", True),
            ("CO2 concentration is now around 524 ppm. This is the highest in at least 2 million years.", True),
            ("Scientific information is taken from ice cores, rocks, and tree rings. These sources provide climate data.", True),
            ("Most of the warming occurred in the past 40 years. The seven most recent years were the warmest.", False),
            ("CO2 is a greenhouse gas. It traps heat in the atmosphere.", True),
            ("More than 50% of youth in the U.S. are worried about climate change. They feel uncertain about the future.", True),
            ("The U.S. has witnessed an increase in intense rainfall events. These events have become more destructive.", True),
            ("Finn's parents usually come to the airport when his flight lands. They enjoy seeing him.", True),
        ]

        results = []
        for sentence, expected in test_cases:
            result = check_pronouns_ollama(sentence, model)
            success = result == expected
            if not success:
                print(f'FAILED: {sentence[:10]}...: Result: {result}. Expected: {expected}')
            else:
                print(f'SUCESS: {sentence[:10]}...: Result: {result}. Expected: {expected}')
            results.append(result)

        assert results == [expected for _, expected in test_cases]




if __name__ == "__main__":
    unittest.main()
