import sys
import re
import ollama
#from transformers import pipeline
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metaphor_detect_pipeline import MetaphorDetector
from metaphor_trans import MetaphorTranslator
import csv
import datetime
from bert_score import score
from sentence_transformers import SentenceTransformer, util, models
import logging

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get script's directory
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)


def load_IMPLI():
    # Initialize an empty dictionary to store the data
    IMPLI = {}
    file = os.path.join(script_dir, "IMPLI_Translation_Data/metaphors/manual_e.tsv")
    # Load data from the first file
    with open(file, 'r', newline='', encoding='utf-8') as f1:
        reader = csv.reader(f1, delimiter='\t')
        for idx, row in enumerate(reader, start=1):
            metaphorical_sentence = row[0].strip()
            literal_sentence = row[1].strip()
            IMPLI[idx] = (metaphorical_sentence, literal_sentence)

    return IMPLI



def get_bertscore(candidate: str, references: list) -> float:
    """
    Computes the BERTScore F1 for a candidate sentence against a list of reference sentences.
    Returns the maximum F1 score across all references.
    """
    _, _, F1 = score([candidate] * len(references), references, lang="en")
    return max(F1).item()  # Take the highest F1 score across references


def get_sbert_similarity(candidate: str, references: list) -> float:
    """
    Computes SBERT cosine similarity for a candidate sentence against a list of references.
    Returns the maximum similarity score across all references.
    """
    cand_embedding = sbert_model.encode(candidate, convert_to_tensor=True)
    ref_embeddings = sbert_model.encode(references, convert_to_tensor=True)
    
    similarities = util.pytorch_cos_sim(cand_embedding, ref_embeddings).squeeze(0)
    return similarities.max().item()  # Return the highest similarity score



data = load_IMPLI()
local_sbert_model_path = os.path.join(script_dir, "../sbert_sentence_transformer")

# Check if model directory exists and has content
if os.path.exists(local_sbert_model_path) and os.listdir(local_sbert_model_path):
    print("Loading SBERT model from local storage...")
    sbert_model = SentenceTransformer(local_sbert_model_path)
else:
    print("Downloading SBERT model from Hugging Face...")
    sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    os.makedirs(local_sbert_model_path, exist_ok=True)
    sbert_model.save(local_sbert_model_path)
    print(f"Model saved locally at {local_sbert_model_path}")

'''Things to try:
   1. Zero-shot approach without context
   2. Zero-shot approach with context from similar sentences in VUA train set
   3. Give the metaphorical token words to ollama and ask to rephrase knowing that those words are metaphorical.'''

# maybe put these in an init class?
mt = MetaphorTranslator('llama3.2')
md = MetaphorDetector()

def foo():
    for i in range(1,4):
        met, lit = data[i]
        # this was bleu score, isn't that great for the task
        # l = [nltk.word_tokenize(lit)]
        # trans = mt.translate_metaphor(met)
        # print(f'Original: {met}   Translation: {trans}')
        # t = nltk.word_tokenize(trans)
        # bleu_score = sentence_bleu(l, t)
        # print(bleu_score)
        trans = mt.translate_metaphor(met)
        if trans == None:
            print('Didn\'t recieve a valid translation. Quitting')
            break
        
        bertscore_f1 = get_bertscore(trans, [lit])
        sbert_similarity = get_sbert_similarity(trans, [lit])
        
        print(f'Original: {met}   Translation: {trans}')
        print(f"BERTScore F1: {bertscore_f1:.4f}")
        print(f"SBERT Cosine Similarity: {sbert_similarity:.4f}")

        met_binary = md.detect_metaphor(met)[0]
        print(f"Metaphor Classification: {met_binary}")



def run_test(num_instances=None):
    '''Translates metaphorical sentences from the IMPLI dataset, calculates the SBERT Cosine Similarity
    and BERTSCORE for the translation compared with the reference sentence, and writes output to a .csv. '''

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = os.path.join(script_dir, f"output/translate_test_output_{now}.csv")

    bertscore_sum = 0
    sbert_sum = 0
    recall_sum = 0

    with open(path, 'w') as out:
        writer = csv.writer(out)
        writer.writerow(['Metaphorical Sentence', 'Literal Sentence', 'Translation', 'BERTScore F1', 'SBERT Cosine Similarity', 'Metaphor Classification'])

        if num_instances != None:
            last = num_instances + 1
        else:
            last = len(data) + 1

        for i in range(1, last):
            met, lit = data[i]
            trans = mt.translate_metaphor(met)
            if trans == None:
                print('Didn\'t recieve a valid translation. Quitting')
                break
            
            bertscore_f1 = get_bertscore(trans, [lit])
            sbert_similarity = get_sbert_similarity(trans, [lit])

            met_binary = md.detect_metaphor(met)[0]

            # Update sums for average calculation
            bertscore_sum += bertscore_f1
            sbert_sum += sbert_similarity
            recall_sum += met_binary

            writer.writerow([met, lit, trans, bertscore_f1, sbert_similarity, met_binary])

        # Calculate averages
        bertscore_avg = bertscore_sum / (last - 1)
        sbert_avg = sbert_sum / (last - 1)
        recall_avg = recall_sum / (last - 1)  # Avoid division by zero

        # Write averages as last row
        writer.writerow(['Averages', '', '', f"{bertscore_avg:.4f}", f"{sbert_avg:.4f}", f"{recall_avg:.4f}"])
run_test()