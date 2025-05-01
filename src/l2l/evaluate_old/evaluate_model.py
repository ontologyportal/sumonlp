from transformers import AutoTokenizer, T5ForConditionalGeneration  # Use T5 classes
import torch
import time

# Example data
predictions = []
references = []

output_file_path = 'predictions_and_references.txt'

def load_model(model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)  # Load your T5 model
    model.eval()  # Set the model to evaluation mode
    return model

def tokenize_input(tokenizer, input_sentence):
    return tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True)

def predict(model, tokenizer, input_sentences):
    pred_time = time.time()
    counter = 0
    predictions = []
    for sentence in input_sentences:
        inputs = tokenize_input(tokenizer, sentence)
        with torch.no_grad():  # Disable gradient calculation
            outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=300)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(decoded_output)
        counter+=1
        if counter % 100 == 0:
          print(f"Counter has reached: {counter} in {time.time() - pred_time:.2f} seconds.")
    return predictions

# Function to calculate exact match
def compute_exact_match(predictions, references):
    # Count matches
    exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred == ref)
    # Calculate percentage
    em_score = exact_matches / len(references) * 100
    return em_score


def main():
    start_time = time.time()
    print('Evaluation started')
    # Load the model and tokenizer from the saved directory
    model = load_model('data/500k_sentences_suffled/t5_model')
    print(f"Model loaded in {time.time() - start_time:.2f} seconds.")
    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    start_time = time.time()
    # Read input sentences for evaluation
    with open('data/500k_sentences_suffled/input_sentences_500k_val.txt', 'r') as f:
        sentences = [line.strip() for line in f.readlines()]
    print(f"Data for evaluation loaded in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    with open('data/500k_sentences_suffled/output_logical_500k_val.txt', 'r') as f:
        references = [line.strip() for line in f.readlines()]
    print(f"Data for reference loaded in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    # Perform inference
    predictions = predict(model, tokenizer, sentences)
    print(f"Predictions completed in {time.time() - start_time:.2f} seconds.")

    start_time = time.time()
    # Calculate exact match score
    exact_match_score = compute_exact_match(predictions, references)
    print(f"Exact Match Score calculated in {time.time() - start_time:.2f} seconds.")
    print("Exact Match Score:", exact_match_score)

        # Write outputs to the output file
    with open(output_file_path, 'w') as f:
        for pred, ref in zip(predictions, references):
            # Write prediction and reference, separated by a tab or comma
            f.write(f"{pred}\t{ref}\n")  # Use \t for tab or use ',' for comma
        f.write(f"Exact Match Score: {exact_match_score}")

if __name__ == "__main__":
    main()
