
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
        
        # get rid of tokenization underscore (_token)
        clean_sentence_metaphors = [token.replace("▁", "") for token in sentence_metaphors]

        sentence_dict['sentence_label'] = sentence_label
        sentence_dict['label_list'] = label_list
        sentence_dict['sentence_metaphors'] =  clean_sentence_metaphors
        print(sentence)
        
        if clean_sentence_metaphors:
            print(clean_sentence_metaphors)
        else:
            print("(No metaphor detected)")

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
            messages = [
                {"role": "system", "content": "You must return only the rephrased sentence without any explanations or additional text."},
                {"role": "user", "content": prompt}
            ]
            response = ollama.chat(model=self.model_type, messages=messages, options={"temperature": 0})
            #response = ollama.chat(model=self.model_type, messages=[{"role": "user", "content": prompt}], options={"temperature": 0})
            return response["message"]["content"].replace('\n', ' ')
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return None

    def replace_words(self, words, sentence):
        
        new_sentence = sentence
        for word in words:
            print(f"Word to replace: ", word)
            prompt = f"Quick answer: For the word {word} in the following sentence: '{sentence}' /\n"
            prompt += f"What is a replacement word I can use that will make the sentence more literal?"
            prompt += "Please respond with ONE replacement word only. Do not include parenthesis. /\n "
            prompt += f"MOST IMPORTANTLY: Your response must be a single word ONLY!!!:\n"

            new_word = self.call_ollama(prompt)
            print(new_word)

            new_sentence = new_sentence.replace(word, new_word)
            print(new_sentence)

        print("FINAL:")
        print(new_sentence)



    def translate_metaphor(self, sentence, sentence_dict=None, include_words=False):
        try:
            if sentence_dict != None and sentence_dict['sentence_label'] == 0:

                return sentence
                
            else:
                if include_words:
                    words = " ".join(sentence_dict['sentence_metaphors'])
                    words = sentence_dict['sentence_metaphors']
                    #print('Words: ', words)
                    #self.replace_words(words, sentence)
                    prompt = f"Quick answer: The sentence: '{sentence}' contains metaphorical words. \n"
                    prompt += f"Those words are: '{words}'. "
                    prompt += "Please rephrase the sentence by replacing ONLY those words with more literal synonomous words. "
                    # prompt += "Return only the sentence with replaced words, keeping the sentence structure about the same. "
                    prompt += "Keep the sentence structure the same, and avoid figurative language. Use the most simple words possible.  "
                    prompt += "Only replace the word in question, do not replace ANY other words in the sentence. "
                    prompt += "Do not add ANY assumptions about the sentence, do not introduce new ideas at all. "
                    prompt += f"MOST IMPORTANTLY: Again, ONLY return the rephrased sentence, no explanations or comments.\n"

                else:
                    prompt = f"Quick answer: Rephrase the following sentence with as few words as possible, without metaphorical content: {sentence}"
                
                print(prompt)
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


