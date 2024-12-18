from transformers import pipeline
import torch

class MetaphorDetector:
    def __init__(self, model_name="lwachowiak/Metaphor-Detection-XLMR"):
        # Set device - 0 for GPU if available, else -1 for CPU
        self.device = 0 if torch.cuda.is_available() else -1
        self.pipe = pipeline("token-classification", model=model_name, device=self.device)

    def detect_metaphor(self, sentence):
        result = self.pipe(sentence)
        label_list = []
        sentence_label = 0
        
        # Extract entities and determine the label
        for dict_entry in result:
            if dict_entry['entity'] == 'LABEL_0':
                label_list.append(0)
            elif dict_entry['entity'] == 'LABEL_1':
                label_list.append(1)
                sentence_label = 1
        
        return sentence_label, label_list

    def process_file(self, input_file, output_file):
        with open(input_file, "r") as infile, open(output_file, "w") as outfile:
            for line in infile:
                sentence = line.strip()
                sentence_label, label_list = self.detect_metaphor(sentence)
                print(f"Sentence: {sentence}")
                print(f"Detected labels: {label_list} -> Overall label: {sentence_label}")
                outfile.write(f"{sentence_label}\t{sentence}\n")


if __name__ == "__main__":
    detector = MetaphorDetector()
    detector.process_file("input_mh.txt", "output_md.txt")
