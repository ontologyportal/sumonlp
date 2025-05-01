from transformers import AutoTokenizer
import json

def tokenize_data(input_file, output_file, tokenized_output_file):
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    # Read the lines from each file
    with open(input_file, 'r') as f_in, open(output_file, 'r') as f_out:
        input_sentences = f_in.readlines()
        output_logical_forms = f_out.readlines()

    # Filter out empty lines
    input_sentences = [line.strip() for line in input_sentences if line.strip()]
    output_logical_forms = [line.strip() for line in output_logical_forms if line.strip()]

    # Tokenize and store the data
    tokenized_data = []
    for input_sentence, output_logical in zip(input_sentences, output_logical_forms):
        input_tokens = tokenizer(input_sentence, return_tensors='pt', padding=True, truncation=True, max_length=512)
        output_tokens = tokenizer(output_logical, return_tensors='pt', padding=True, truncation=True, max_length=512)

        # print(f"Input shape: {input_tokens['input_ids'].shape}, Output shape: {output_tokens['input_ids'].shape}")

        tokenized_data.append({
            'input_ids': input_tokens['input_ids'].tolist(),
            'attention_mask': input_tokens['attention_mask'].tolist(),
            'output_ids': output_tokens['input_ids'].tolist(),
            'output_attention_mask': output_tokens['attention_mask'].tolist()
        })

    # Save the tokenized data to a JSON file
    with open(tokenized_output_file, 'w') as f:
        json.dump(tokenized_data, f)

    print(f"Tokenized dataset saved to {tokenized_output_file}")

# Example usage
# tokenize_data('input_sentences_500k.txt', 'output_logical_500k.txt', 'tokenized_data_small.json')
