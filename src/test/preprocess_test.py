import pandas as pd
import unittest
import sys
import re
import ollama
#from transformers import pipeline
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu
import os
import csv
import datetime
from bert_score import score
from sentence_transformers import SentenceTransformer, util, models
import logging
import subprocess
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
#import classes/functions from metaphor handler, sentence simplification
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
met_dir = os.path.join(script_dir, '..', 'metaphor_handling')
sys.path.append(met_dir)
from metaphor_detect_pipeline import MetaphorDetector
from metaphor_trans import MetaphorTranslator

oov_directory = os.path.join(script_dir, '..', 'oov_handling')
sys.path.append(oov_directory)
import oov_handling

ss_dir = os.path.join(script_dir, '..', 'sentence_simplification')
sys.path.append(ss_dir)
from simplification import simplify_file, simplify_sentence_llm

print("Done importing.")

# Helper function to load CSV into DataFrame

def load_excel(file_path=os.path.join(script_dir, "Integration_Test_Set.xlsx")):
    return pd.read_excel(file_path)
# Unit Test Class

results = pd.DataFrame(columns=[
                                    "Original Sentence",
                                    "Metaphor Label",
                                    "Metaphor Detected",
                                    "Metaphor Resolution",
                                    "Metaphor Translated",
                                    "Original-MT BERTScore",
                                    "Original-MT SBERT CosineSim",
                                    "Sentence Simplification",
                                    "Sentence Simplification Results",
                                    "SS-MR BERTScore",
                                    "SS-MR SBERT CosineSim",
                                    "Out of Vocab Handling",
                                    "OOV Output"
                                    ])

class TestPipeline(unittest.TestCase):

    
    def setUp(self):
        """Set up before each test."""
        self.df = load_excel()
        self.md = MetaphorDetector()
        self.mt = mt = MetaphorTranslator('llama3.2')
        # self.results = pd.DataFrame(columns=[
        #                             "Original Sentence",
        #                             "Metaphor Label",
        #                             "Metaphor Detected",
        #                             "Metaphor Resolution",
        #                             "Metaphor Translated",
        #                             "Original-MT BERTScore",
        #                             "Original-MT SBERT CosineSim",
        #                             "Sentence Simplification",
        #                             "Sentence Simplification Results",
        #                             "SS-MR BERTScore",
        #                             "SS-MR SBERT CosineSim",
        #                             "Out of Vocab Handling",
        #                             "OOV Output"
        #                             ])

        local_sbert_model_path = os.path.join(script_dir, "../metaphor_handling/sbert_sentence_transformer")
        #/home/jarrad.singley/data/workspace/sumonlp/src/metaphor_handling/sbert_sentence_transformer
        #/home/jarrad.singley/data/workspace/sumonlp/test
        # Check if model directory exists and has content
        if os.path.exists(local_sbert_model_path) and os.listdir(local_sbert_model_path):
            print("Loading SBERT model from local storage...")
            self.sbert_model = SentenceTransformer(local_sbert_model_path)
        else:
            print("Downloading SBERT model from Hugging Face...")
            self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
            os.makedirs(local_sbert_model_path, exist_ok=True)
            self.sbert_model.save(local_sbert_model_path)
            print(f"Model saved locally at {local_sbert_model_path}")

    def test_metaphor_detection(self):
        """
        Test Metaphor Detection
        - Need to convert to .sh run + file IO like done in test_OOV_handling
        """
        for idx, row in self.df.iterrows():
            sentence = row['Original Sentence']
            flag = 1 if row["Metaphor Label"] == "Y" else 0
            results.loc[idx, 'Metaphor Label'] = row["Metaphor Label"]
            results.loc[idx, 'Original Sentence'] = sentence

            # Test logic for metaphor detection should go here
            result = self.md.detect_metaphor(sentence)[0]
            results.loc[idx, 'Metaphor Detected'] = result

            with self.subTest(idx=idx, sentence=sentence):
                self.assertEqual(flag, result)
        print(results)
            

    def test_metaphor_resolution(self):
        """
        Test Metaphor Resolution
        - Need to convert to .sh run + file IO like done in test_OOV_handling        
        """
        for idx, row in self.df.iterrows():
            sentence = row['Original Sentence']
            expected_resolution = row['Metaphor Resolution']
            results.loc[idx, "Metaphor Resolution"] = expected_resolution

            with self.subTest(idx=idx, sentence=sentence):

                met_flag = results.loc[idx, "Metaphor Detected"]
                if met_flag:
                    trans = self.mt.translate_metaphor(sentence)
                    bertscore_f1 = self.get_bertscore(trans, [expected_resolution])
                    sbert_similarity = self.get_sbert_similarity(trans, [expected_resolution])
                    self.assertGreater(bertscore_f1, 0.80)
                    self.assertGreater(sbert_similarity, 0.80)
                else:
                    trans = sentence
                    bertscore_f1 = 'n/a'
                    sbert_similarity = 'n/a'



                results.loc[idx, "Metaphor Translated"] = trans
                results.loc[idx, "Original-MT BERTScore"] = bertscore_f1
                results.loc[idx, "Original-MT SBERT CosineSim"] = sbert_similarity


        print(results)
        self.write_out()

    def test_sentence_simplification(self):
        """
        Test Sentence Simplification
        - Need to convert to .sh run + file IO like done in test_OOV_handling
        """
        temp = os.path.join(script_dir, 'temp.txt')  # fix later so we don't need to do file I/O for SS
        print("IN SENTENCE SIMPLIFICATION")
        for idx, row in self.df.iterrows():
            sentence = row['Metaphor Resolution']
            expected_simplification = row['Sentence Simplification']
            results.loc[idx, "Sentence Simplification"] = expected_simplification
            # Test logic for sentence simplification

            with self.subTest(idx=idx, sentence=sentence):
                with open(temp, 'w', encoding='utf-8') as t:
                    t.write(sentence)
                dt = datetime.datetime.now().strftime('%d%b%Y_%H%M').upper()

                result = simplify_file(temp, dt)[0]  # simplify file returns a list of output sentences, should only be 1 in this list

                bertscore_f1 = self.get_bertscore(result, [expected_simplification])
                sbert_similarity = self.get_sbert_similarity(result, [expected_simplification])

                results.loc[idx, "Sentence Simplification Results"] = result
                results.loc[idx, "SS-MR BERTScore"] = bertscore_f1
                results.loc[idx, "SS-MR SBERT CosineSim"] = sbert_similarity

                self.assertGreater(bertscore_f1, 0.80)
                self.assertGreater(sbert_similarity, 0.80)
        print(results)
        self.write_out()
    
    def test_out_of_vocab_handling(self):
        """
        Test Out of Vocab Handling
        """
        input = os.path.join(script_dir, '..', 'src', 'oov_handling', 'input_oov.txt')
        output = os.path.join(script_dir, '..', 'src', 'oov_handling', 'output_oov.txt')

        # copy expected OOV output to new df
        for idx, row in self.df.iterrows():
            expected_oov = row['Out of Vocab Handling']
            results.loc[idx, 'Out of Vocab Handling'] = expected_oov

        with open(input, 'w', encoding='utf-8') as infile:
            for line in results['Sentence Simplification']:
                infile.write(line + '\n')
        

        # run the OOV script
        script_path = oov_directory + '/entry_point.sh'
        run_oov = subprocess.run(['bash', script_path], capture_output=True, text=True)

        with open(output, 'r', encoding='utf-8') as outfile:
            lines = outfile.readlines()

        oov_lines = lines[1::2]

        # Make sure the output has the same number of lines as the 'Sentence Simplification' column
        #assert len(oov_lines) == len(results), "Mismatch between OOV output lines and DataFrame rows"

        # Iterate over each line in the output and insert it into the DataFrame
        for idx, line in enumerate(oov_lines):
            results.loc[idx, 'OOV Output'] = line.strip()  # Remove any extra newlines

        self.write_metrics()
        self.write_out()

    def get_bertscore(self, candidate: str, references: list) -> float:
        """
        Computes the BERTScore F1 for a candidate sentence against a list of reference sentences.
        Returns the maximum F1 score across all references.
        """
        _, _, F1 = score([candidate] * len(references), references, lang="en")
        return max(F1).item()  # Take the highest F1 score across references


    def get_sbert_similarity(self, candidate: str, references: list) -> float:
        """
        Computes SBERT cosine similarity for a candidate sentence against a list of references.
        Returns the maximum similarity score across all references.
        """
        cand_embedding = self.sbert_model.encode(candidate, convert_to_tensor=True)
        ref_embeddings = self.sbert_model.encode(references, convert_to_tensor=True)
        
        similarities = util.pytorch_cos_sim(cand_embedding, ref_embeddings).squeeze(0)
        return similarities.max().item()  # Return the highest similarity score

    def write_metrics(self):
        global results  # Ensure we're modifying the global DataFrame

        # Convert "n/a" or non-numeric values to NaN so that mean calculation ignores them
        bert_sbert_cols = [
            "Original-MT BERTScore",
            "Original-MT SBERT CosineSim",
            "SS-MR BERTScore",
            "SS-MR SBERT CosineSim"
        ]

        for col in bert_sbert_cols:
            results[col] = pd.to_numeric(results[col], errors='coerce')

        # Compute mean for BERT/SBERT scores (ignoring NaNs)
        bert_sbert_means = results[bert_sbert_cols].mean()

        # Convert 'Metaphor Label' to binary (1 = 'Y', 0 = 'N') and ensure 'Metaphor Detected' is int
        valid_rows = results["Metaphor Detected"].notna()  # Identify valid rows
        y_true = results.loc[valid_rows, "Metaphor Label"].map({"Y": 1, "N": 0})
        y_pred = results.loc[valid_rows, "Metaphor Detected"].astype(int)

        # Compute accuracy, precision, and recall
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        # Create a structured metrics summary
        metrics_rows = [
            {"Original Sentence": "Metrics"},  # Header row
            {**{"Original Sentence": "BERT/SBERT Score Averages"}, **bert_sbert_means.to_dict()},  # BERT/SBERT scores
            {"Original Sentence": "Metaphor Detection Accuracy", "Metaphor Label": accuracy},
            {"Original Sentence": "Metaphor Detection Precision", "Metaphor Label": precision},
            {"Original Sentence": "Metaphor Detection Recall", "Metaphor Label": recall}
        ]

        # Append the rows to results DataFrame
        results = pd.concat([results, pd.DataFrame(metrics_rows)], ignore_index=True)


    def write_out(self):
        print('In write out')
        out = os.path.join(script_dir, 'output.csv')
        results.to_csv(out, index=False)


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestPipeline('test_metaphor_detection'))
    suite.addTest(TestPipeline('test_metaphor_resolution'))  # Add tests in desired order
    suite.addTest(TestPipeline('test_sentence_simplification'))
    suite.addTest(TestPipeline('test_out_of_vocab_handling'))

    runner = unittest.TextTestRunner(failfast=False)
    runner.run(suite)

    # test_pipeline_instance = TestPipeline()  # Ensure the instance is created before calling write_out
    # test_pipeline_instance.write_out()  # Call write_out method on the instance