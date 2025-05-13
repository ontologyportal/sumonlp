import os
import torch
import sys
import argparse
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, PreTrainedTokenizerFast
import json
import time

from transformers import LogitsProcessorList, LogitsProcessor

# Define the custom processor
class ForceAllowedTokensProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = set(allowed_token_ids)

    def __call__(self, input_ids, scores):
        # Create a mask for allowed tokens, set all others to -inf
        mask = torch.full_like(scores, float("-inf"))

        # Allow only the allowed token IDs
        for token_id in self.allowed_token_ids:
            mask[:, token_id] = scores[:, token_id]

        return mask


def main():
    filepath = sys.argv[1] + "t5_model/"

    # Load the input tokenizer (standard T5)
    input_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    # Load the output tokenizer (custom)
    output_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=filepath+"tokenizer.json",
        tokenizer_config_file=filepath+"tokenizer_config.json",
        special_tokens_map_file=filepath+"special_tokens_map.json"
    )

    # Load the trained model
    model = T5ForConditionalGeneration.from_pretrained(
        filepath,
        ignore_mismatched_sizes=True
    )
    model.eval()

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Read input sentences from the input file
    with open('./input_l2l.txt', 'r') as f:
        sentences = [line.strip() for line in f.readlines()]

    # Perform inference
    predictions = []
    for sentence in sentences:
        if(not sentence.startswith('SentenceId:')):
            # Tokenize the input
            inputs = input_tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():  # Disable gradient calculation
                outputs = model.generate(
                    **inputs,
                    max_length=128,
                    num_beams=4,
                    early_stopping=True
                )

            decoded_output = output_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print("Decoded output: " + str(decoded_output))
            predictions.append(decoded_output)
        else:
            predictions.append(sentence)

    # Write outputs to the output file
    with open('./output_l2l.txt', 'w') as f:
        for sentence, prediction in zip(sentences, predictions):
            f.write(f"{prediction}\n")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Processing complete in {time.time() - start_time:.2f} seconds.")
