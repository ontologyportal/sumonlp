
from transformers import pipeline
import torch
import sys
import re
import ollama
import time

class MetaphorDetector:
    def __init__(self, model_name="lwachowiak/Metaphor-Detection-XLMR"):
        # Set device - 0 for GPU if available, else -1 for CPU
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("token-classification", model=model_name, device=self.device)

    def detect_metaphor(self, sentence):
        result = self.pipe(sentence)
        label_list = []   # a list of binary classification labels for all tokens in the sentence
        sentence_label = 0  # initialized binary classification label
        sentence_metaphors = []  # tokens in a sentence that have been classified as a metaphor
        sentence_dict = {}     # a dictionary to store all of the above and return as an output as either a list of dicts (in process file) or a single dict (for just one sentence)
        
        # Extract entities and determine the label
        for dict_entry in result:
            if dict_entry['entity'] == 'LABEL_0':
                label_list.append(0)
            elif dict_entry['entity'] == 'LABEL_1':
                label_list.append(1)
                sentence_label = 1
                sentence_metaphors.append(dict_entry['word'])
        
        sentence_dict['sentence_label'] = sentence_label
        sentence_dict['label_list'] = label_list
        sentence_dict['sentence_metaphors'] =  sentence_metaphors

        return sentence, sentence_dict

    def process_file(self, input_file):
        file_results = {}
        with open(input_file, "r") as infile:
            for line in infile:
                sentence = line.strip()
                sentence, sentence_dict = self.detect_metaphor(sentence)
                file_results[sentence] = sentence_dict
        
        return file_results




class MetaphorTranslator:

    def __init__(self, model_type: str):
        # Initialize model type
        self.model_type = model_type



    def call_ollama(self, prompt: str):
        try:
            response = ollama.chat(model=self.model_type, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
            return response["message"]["content"].replace('\n', ' ')
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None

    def translate_metaphor(self, sentence, sentence_dict=None, include_words=False):
        try:
            if sentence_dict != None and sentence_dict['sentence_label'] == 0:

                return sentence
                
            else:
                if include_words:
                    words = "".join(sentence_dict['sentence_metaphors'])
                    prompt = f"Quick answer: This sentence contains metaphorical content: {sentence} /\n"
                    prompt += f"The following words in the sentence may be metaphorical: {words}. "
                    prompt += "Please rephrase the sentence by replacing the words and ensure your response is free of metaphorical words or expressions."
                    prompt += "Please return ONLY the rephrased sentence in your answer."
                else:
                    prompt = f"Quick answer: Rephrase the following sentence with as few words as possible, without metaphorical content: {sentence}"
                
                response = self.call_ollama(prompt)

                if response != None:
                    response = re.sub(r'\s*\([^)]*\)\s*$', '.', response)  # Removes trailing parentheticals
                
                return response

        except Exception as e:
            print(f"Error, could not translate metaphor: {e}")
            return None

    def process_file(self, detection_results, output_file, include_words=False):
        with open(output_file, 'w') as outfile:
            for sentence, sentence_dict in detection_results.items():
                translation = self.translate_metaphor(sentence, sentence_dict, include_words).strip()
                outfile.write(translation + '\n')



# metaphor detector main
# if __name__ == "__main__":
#     detector = MetaphorDetector()
#     detector.process_file("input_mh.txt", "output_md.txt")

# metaphor translator main
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file> <output_file> <model_type>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    model_type = sys.argv[3]
    detector = MetaphorDetector()
    translator = MetaphorTranslator(model_type=model_type)
    print("Detecting metaphors...")
    detection_results = detector.process_file(input_file)
    print("Translating metaphors...")
    translator.process_file(detection_results, output_file, include_words=True)  # change this if you just want to give ollama the sentence without detected met. words
    print(f"Processing complete. Responses saved to {output_file}.")


