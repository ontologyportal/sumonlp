import json
import torch
import os
from transformers import AutoTokenizer
import jsonlines

class TokenizedDatasetPreprocessor:
    def __init__(self, **config):

        self.save_path = config["save_path"]  # Where to save preprocessed data
        self.data_name = config["data_name"]

        self.input_file = config["input_file"]

        self.label_file = config["label_file"]

        self.tokenizer_name = config["tokenizer_name"]
        self.max_input_length = config.get("max_input_length", 512)
        self.max_output_length = config.get("max_output_length", 512)

        # Load tokenizer
        print(f"Initializing tokenizer: {self.tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)


    def preprocess_and_save(self):

          # input_filename = os.path.basename(self.input_file)
          # input_filename_no_ext = os.path.splitext(input_filename)[0]
          tokenized_output_file = os.path.join(self.save_path, self.data_name+".jsonl")
          index_output_file = os.path.join(self.save_path, self.data_name + ".index")


          if os.path.exists(tokenized_output_file) and os.path.exists(index_output_file):
              print(f"Tokenized Data & Index Already Exist! Skipping.")
              return  # Exit if files already exist
          else:
              print(f"Starting new tokenization!")

          # Read input JSON file
          with open(self.input_file, "r", encoding="utf-8") as f_in:
              data = json.load(f_in)

          # Initialize lists
          input_sentences = []
          output_logical_forms = []

          prefix = "Convert the following sentence to SUO-KIF: "

          for entry in data:
            input_sentences.append(prefix + entry["input"].strip())
            output_logical_forms.append(entry["output"].strip())

          os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

          # Open files for writing
          offsets = []
          with open(tokenized_output_file, mode='w', encoding="utf-8") as f:
              for input_sentence, output_logical in zip(input_sentences, output_logical_forms):
                  offset = f.tell()  # Store the current byte offset
                  input_tokens = self.tokenizer(
                      input_sentence, return_tensors='pt', padding=True, truncation=True, max_length=self.max_input_length
                  )
                  output_tokens = self.tokenizer(
                      output_logical, return_tensors='pt', padding=True, truncation=True, max_length=self.max_output_length
                  )

                  sample = {
                      'input_ids': input_tokens['input_ids'].tolist(),
                      'attention_mask': input_tokens['attention_mask'].tolist(),
                      'output_ids': output_tokens['input_ids'].tolist(),
                  }

                  json.dump(sample, f)
                  f.write("\n")  # Ensure each sample is a new line
                  offsets.append(offset)

          with open(index_output_file, "w") as index_writer:
            json.dump(offsets, index_writer)

          print(f"Tokenized dataset saved to {tokenized_output_file}")
          print(f"Index file saved to {index_output_file}")
