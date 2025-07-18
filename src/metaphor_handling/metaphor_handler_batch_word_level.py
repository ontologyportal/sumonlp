
from transformers import pipeline
import torch
import sys
import re
import ollama
import time
import json
import math
from collections import defaultdict
from utils import *

class MetaphorDetector:
    def __init__(self, model_name="lwachowiak/Metaphor-Detection-XLMR"):
        # Set device - 0 for GPU if available, else -1 for CPU
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("token-classification", model=model_name, device=self.device)




    def detect_metaphor(self, sentence):
        result = self.pipe(sentence)
        label_list = []
        sentence_label = 0
        raw_mets = {}
        for dict_entry in result:
            if dict_entry['entity']=='LABEL_1':
                label_list.append(1)
                sentence_label = 1
                raw_mets[dict_entry['start']] = dict_entry['word'].replace("▁","")
            else:
                label_list.append(0)

        sentence_dict = {
            'sentence': sentence,
            'sentence_label': sentence_label,
            'label_list': label_list,
            'sentence_metaphors': raw_mets
        }

        # now let your fixer strip out punctuation and overlapping fragments
        sentence_dict['sentence_metaphors'] = fix_metaphor_words_dict(sentence_dict)
        return sentence_dict

    def process_file(self, input_file):
        file_results = []
        with open(input_file, "r") as infile:
            for line in infile:
                sentence = line.strip()
                sentence_dict = self.detect_metaphor(sentence)
                file_results.append(sentence_dict)
        
        return file_results

class MetaphorTranslatorBatchWordLevel:
    def __init__(
        self,
        model_type: str='llama3.1:8b',
        similarity_function: str='bert_cosine',
        bert_embedder: BertEmbedder | None = None,
        md: MetaphorDetector | None = None,
        start_temp: float=0.2,
        desired_sim_score: float=0.8,
        prompt_limit: int=3,
        batch_size: int=6,
        reduction_rate: float=0.5,
    ):
        self.model_type = model_type
        self.similarity_function = similarity_function
        self.temp = start_temp # no dynamic temperature changing for this class
        self.desired_sim_score = desired_sim_score
        self.prompt_limit = prompt_limit
        self.batch_size = batch_size
        self.reduction_rate = reduction_rate

        # Bert embedder injection
        if isinstance(bert_embedder, BertEmbedder):
            self.bert_embedder = bert_embedder
        else:
            self.bert_embedder = BertEmbedder(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                use_mean_pooling=True
            )

        # Metaphor detector injection
        if isinstance(md, MetaphorDetector):
            self.md = md
        else:
            self.md = MetaphorDetector()

        ## Chapter 4 ##
        # here are some class vars to store things about the translation process
        self.candidates = {}
        self.prompt_attempts = defaultdict(int)
        self.translations = {}
        self.selection_criteria = {}

        # these count the criteria that the final selected translations meet
        self.passed_both = 0
        self.passed_sim = 0
        self.passed_met = 0
        self.passed_neither = 0

    def call_ollama(self, prompt: str):
        try:
            messages = [
                {"role": "system", "content": "You must return only a JSON object with the specified fields and no additional explanations or text."},
                {"role": "user", "content": prompt}
            ]
            response = ollama.chat(model=self.model_type, messages=messages, options={"temperature": self.temp})
            # Replace newlines with spaces to ensure a single-line JSON output.
            return response["message"]["content"].replace('\n', ' ')
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None

    def extract_json_output(self, json_str):
        # grab the {...} block
        m = re.search(r'\{.*\}', json_str, flags=re.DOTALL)
        if not m:
            print("No JSON found.")
            return None
        blob = m.group(0)

        # Fix missing commas between values and new keys
        blob = re.sub(r'(".*?")\s*("(\d+)":)', r'\1, \2', blob)
        # Remove trailing commas before closing braces/brackets
        blob = re.sub(r',(\s*[}\]])', r'\1', blob)
        try:
            data = json.loads(blob)
            return {k:v for k,v in data.items() if k.isdigit() and isinstance(v, str)}
            # could return an empty dict
        except json.JSONDecodeError as e:
            print("Cleaned JSON still invalid:", e)
            print("=>", blob)
            return None


    def extract_json_output_old(self, json_str):
        """
        Extracts all reworded sentences from a JSON string output.
        
        The JSON is expected to have keys:
        - "original": the original metaphorical sentence
        - "1", "2", ..., "n": each corresponding to a reworded translation
        
        Args:
            json_str (str): A JSON-formatted string with the specified fields.
        
        Returns:
            dict: A dictionary mapping translation numbers (as strings) to their reworded sentences.
                For example: {"1": "Reworded sentence one.", "2": "Reworded sentence two.", ...}
                Returns an empty dictionary if no translations are found.
        """
        try:
            data = json.loads(json_str)
            # Build a dictionary only for keys that are digits, which represent reworded sentences.
            translations = {key: value for key, value in data.items() if key.isdigit()}
            return translations
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return None

    def _extract_with_retries(self, prompt: str, max_retries: int = 5):
        """
        Call Ollama up to max_retries times, extracting JSON each time.
        Returns the first non‑None JSON dict or None if all retries fail.
        """
        for attempt in range(1, max_retries + 1):
            raw = self.call_ollama(prompt)
            data = self.extract_json_output(raw)
            if data:
                return data
            print(f"Retry {attempt}: Failed to extract JSON output")
        return None

    def translate_metaphor(self, sentence_dict=None, sentence_label=None):

        if sentence_dict is None:
            raise ValueError("translate_metaphor: 'sentence_dict' is required")

        sentence = sentence_dict['sentence']
        orig_metaphors = sentence_dict['sentence_metaphors'].values()
        og_count = len(orig_metaphors)

        # compute max allowed metaphors in a candidate
        max_allowed = math.floor(og_count * (1 - self.reduction_rate))

        accepted = []             # within tolerance & ≥ threshold
        tolerance_candidates = [] # within tolerance, any score
        overall = []              # all candidates: (resp, score, count)

        ## Chapter 4 ##
        if sentence_label:
            self.prompt_attempts[sentence_label] = 0
        ################

        for attempt in range(self.prompt_limit):

            prompt = gen_batch_translation_prompt(
                sentence,
                self.batch_size,
                follow_on=bool(attempt),
                flagged_words=orig_metaphors
            )

            responses = self._extract_with_retries(prompt, max_retries=5)

            ## Chapter 4 ##
            if sentence_label:
                self.prompt_attempts[sentence_label] += 1
            ################
            if responses is None:
                print(f"Retry {attempt}: no JSON, skipping")
                continue

            for resp in responses.values():
                # similarity
                score = compute_similarity(
                    self.similarity_function,
                    sentence,
                    resp,
                    bert_embedder=getattr(self, 'bert_embedder', None)
                )

                # redetect metaphors
                det = self.md.detect_metaphor(resp)
                trans_count = len(det['sentence_metaphors'].values())

                overall.append((resp, score, trans_count))

                # check tolerance
                if trans_count <= max_allowed:
                    tolerance_candidates.append((resp, score, trans_count))
                    if score >= self.desired_sim_score:
                        accepted.append((resp, score, trans_count))
                        print('★' if trans_count == 0 else '☆', end='', flush=True)

                    else:
                        print('✔', end='', flush=True)
                else:
                    print('•' if score >= self.desired_sim_score else '.', end='', flush=True)

            print(' ', end='', flush=True)
            if accepted:
                break

        print('\n')

        # pick best
        sel_criteria = None
        if accepted:
            best_resp, best_score, best_count = max(accepted, key=lambda x: x[1])
            self.passed_both += 1
            sel_criteria = (True, True)
        elif tolerance_candidates:
            best_resp, best_score, best_count = max(tolerance_candidates, key=lambda x: x[1])
            self.passed_met += 1
            sel_criteria = (True, False)
        elif overall:
            best_resp, best_score, best_count = max(overall, key=lambda x: x[1])
            if best_score >= self.desired_sim_score:
                self.passed_sim += 1
                sel_criteria = (False, True)
            else:
                self.passed_neither += 1
                sel_criteria = (False, False)

        else:
            best_resp, best_score, best_count = "", 0.0, 0
            sel_criteria = (None, None)

        if sentence_label and overall:
            self.candidates[sentence_label] = overall
            self.translations[sentence_label] = (best_resp, best_score, best_count)
        
        if sentence_label:
            self.selection_criteria[sentence_label] = sel_criteria

        return best_resp



    def process_file(self, detection_results, output_file):
        with open(output_file, 'w') as outfile:
            for i, sentence_dict in enumerate(detection_results, start=1):
                sentence = sentence_dict['sentence']
                print(f'Sentence {i}:  "{sentence}"')
                if sentence_dict['sentence_label'] == 0:
                    outfile.write(sentence + '\n')
                    continue
                translation = self.translate_metaphor(sentence_dict, sentence_label=i).strip()
                while translation == "":
                    print('Running translate_metaphor again...')
                    translation = self.translate_metaphor(sentence_dict, sentence_label=i).strip()
                outfile.write(translation + '\n')



# metaphor detector main
# if __name__ == "__main__":
#     detector = MetaphorDetector()
#     detector.process_file("input_mh.txt", "output_md.txt")

# metaphor translator main
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    md = MetaphorDetector()
    translator = MetaphorTranslatorBatchWordLevel(
            model_type='llama3.1:8b',
            similarity_function='bleu',
            md=md,
            start_temp=0.2,
            desired_sim_score=0.1,
            prompt_limit=4,
            batch_size=10,
            reduction_rate=0.7     ### testing here
        )
    print("Detecting metaphors...")
    detection_results = md.process_file(input_file)
    #print(detection_results)
    print("Translating metaphors...")
    translator.process_file(detection_results, output_file)  # change this if you just want to give ollama the sentence without detected met. words
    print(f"Processing complete. Responses saved to {output_file}.")


