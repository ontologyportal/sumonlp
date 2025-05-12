import os
import torch
import sys
import argparse
from transformers import AutoTokenizer, T5ForConditionalGeneration, T5Tokenizer, PreTrainedTokenizerFast
import json
import time

from transformers import LogitsProcessorList, LogitsProcessor

class ForceAllowedTokensProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        # Filter out None values and ensure all are integers
        cleaned_token_ids = [int(tid) for tid in allowed_token_ids if tid is not None]
        # Convert to tensor for much faster operations
        self.allowed_token_ids_tensor = torch.tensor(cleaned_token_ids, dtype=torch.long)
        print(f"Number of allowed tokens (after cleaning): {len(cleaned_token_ids)}")

    def __call__(self, input_ids, scores):
        # Create a new tensor of negative infinity with the same shape as scores
        new_scores = torch.full_like(scores, float("-inf"))

        # Only set scores for allowed tokens
        # This is much more efficient than iterating through each allowed token
        new_scores[:, self.allowed_token_ids_tensor] = scores[:, self.allowed_token_ids_tensor]

        return new_scores

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

    # Print vocabulary size information
    print(f"Input tokenizer vocabulary size: {input_tokenizer.vocab_size}")
    print(f"Output tokenizer vocabulary size: {output_tokenizer.vocab_size}")

    # Load the trained model
    model = T5ForConditionalGeneration.from_pretrained(
        filepath,
        ignore_mismatched_sizes=True
    )
    model.eval()

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)

    # Read input sentences from the input file
    with open('./input_l2l.txt', 'r') as f:
        sentences = [line.strip() for line in f.readlines()]

    print(f"Loaded {len(sentences)} sentences for processing")

    # Build list of allowed token IDs
    allowed_token_ids = set()

    # ALWAYS include special tokens
    special_tokens = {
        output_tokenizer.pad_token_id,
        output_tokenizer.eos_token_id,
        output_tokenizer.bos_token_id,
        output_tokenizer.unk_token_id  # Sometimes needed for generation
    }
    # Make sure to only add non-None values
    allowed_token_ids.update([t for t in special_tokens if t is not None])

    # Load the tokenizer.json file
    with open(filepath+"tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer_data = json.load(f)

    # Extract allowed tokens from the JSON data
    # For tokenizers using "added_tokens"
    if "added_tokens" in tokenizer_data:
        print("Processing added_tokens from tokenizer data")
        for token in tokenizer_data.get("added_tokens", []):
            if not token.get("special", False):
                token_id = token.get("id")
                if token_id is not None:
                    allowed_token_ids.add(int(token_id))  # Ensure it's an integer

    '''
    # For tokenizers using "model.vocab" as a nested list structure
    if "model" in tokenizer_data and "vocab" in tokenizer_data["model"]:
        print("Processing vocab from tokenizer model data")
        vocab = tokenizer_data["model"]["vocab"]

        print(f"Vocab type: {type(vocab)}")
        if isinstance(vocab, dict):
            # If it's a dictionary (key-value pairs)
            for token, idx in vocab.items():
                if idx is not None:
                    allowed_token_ids.add(int(idx))  # Ensure it's an integer
        elif isinstance(vocab, list):
            # For nested list format: [['token', score], ...]
            if vocab and isinstance(vocab[0], list) and len(vocab[0]) > 1:
                print(f"First vocab item type: {type(vocab[0])}")
                print(f"First few vocab items: {vocab[:5]}")
                # Extract token IDs based on position in the list
                for i, (token, _) in enumerate(vocab):
                    allowed_token_ids.add(i)
    '''
    # Explicitly define common structural tokens that should be allowed
    structural_tokens = ['"', '"▁', '▁"', '(', ')', '▁', '▁)', ')▁','(▁', '▁(',' ', '\t', '\n', '\r']
    for token in structural_tokens:
        token_id = output_tokenizer.convert_tokens_to_ids(token)
        if token_id != output_tokenizer.unk_token_id or token.isspace():
            allowed_token_ids.add(token_id)

    # If we can't extract enough tokens, try a different approach
    if len(allowed_token_ids) < 100:
        print("Few tokens detected. Trying direct vocab extraction...")

        # Try to access the vocabulary directly from the tokenizer
        if hasattr(output_tokenizer, 'get_vocab'):
            vocab_dict = output_tokenizer.get_vocab()
            for token_id in vocab_dict.values():
                if token_id is not None:
                    allowed_token_ids.add(int(token_id))

        # If still too few tokens, include all possible token IDs
        if len(allowed_token_ids) < 100:
            print("WARNING: Still few allowed tokens. Including all tokens in vocabulary range.")
            allowed_token_ids.update(range(output_tokenizer.vocab_size))

    # Make sure we have only valid integers
    allowed_token_ids = {int(tid) for tid in allowed_token_ids if tid is not None}

    print(f"Total allowed token IDs: {len(allowed_token_ids)}")
    print(f"Sample allowed token IDs: {list(allowed_token_ids)[:10]}")

    # Sample a few allowed tokens to verify they make sense
    sample_ids = list(allowed_token_ids)[:10000]
    sample_tokens = output_tokenizer.convert_ids_to_tokens(sample_ids)
    print("Sample allowed tokens:")
    for i, token in zip(sample_ids, sample_tokens):
        print(f"ID: {i}, Token: {repr(token)}")

    # Build logits processor list
    logits_processor = LogitsProcessorList([
        ForceAllowedTokensProcessor(allowed_token_ids)
    ])

    # Perform inference
    predictions = []
    for idx, sentence in enumerate(sentences):
        if not sentence.startswith('SentenceId:'):
            print(f"\n[{idx+1}/{len(sentences)}] Processing: {sentence[:50]}...")

            # Tokenize the input
            inputs = input_tokenizer(
                sentence,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            try:
                # Generate with token restrictions
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=128,
                        num_beams=4,
                        early_stopping=True,
                        logits_processor=logits_processor,
                        do_sample=False
                    )

                # Decode output tokens
                decoded_output = output_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                print(f"Output: {decoded_output}..." if decoded_output else "Empty output")

                # Check the first few output tokens
                #first_output = outputs.cpu().numpy()
                #first_tokens = output_tokenizer.convert_ids_to_tokens(first_output)
                #print(f"First output tokens: {first_tokens}")

                predictions.append(decoded_output[0])

            except Exception as e:
                print(f"Error during generation: {e}")
                print("Using empty string as fallback")
                predictions.append("")
        else:
            predictions.append(sentence)

    # Write outputs to the output file
    with open('./output_l2l.txt', 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Processing complete in {time.time() - start_time:.2f} seconds.")