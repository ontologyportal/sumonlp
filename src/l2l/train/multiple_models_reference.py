import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration

def load_model(model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)  # Load your T5 model
    model.eval()  # Set the model to evaluation mode
    return model

def tokenize_input(tokenizer, input_sentence):
    return tokenizer(input_sentence, return_tensors="pt", padding=True, truncation=True)

def predict(model, tokenizer, input_sentences):
    predictions = []
    for sentence in input_sentences:
        inputs = tokenize_input(tokenizer, sentence)
        with torch.no_grad():  # Disable gradient calculation
            outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_new_tokens=300)
        decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(decoded_output)
    return predictions

def main():
    # List of model paths
    model_paths_names = [
      ('data/full_12m_sentences/t5_model_3_epochs','12m Sentences'),
      ('data/500k_sentences_suffled/t5_model','500k SUFFLED Sentences'),
      ('data/500k_last_sentences/t5_model','500k Last Sentences')
    ]  # Add your actual model paths here

    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    # Load all models once
    models = [(name, load_model(path)) for path, name in model_paths_names]
    # models = [(path, load_model(path)) for path in model_paths_names[0]]

    # Read input sentences from the input file
    with open('./input.txt', 'r') as f:
        sentences = [line.strip() for line in f.readlines()]

    # Write outputs to the output file
    with open('./output_multiple_models.txt', 'w') as f:
        for sentence in sentences:
            f.write(f"Input: {sentence}\n")
            for model_path, model in models:
                model_name = model_path.split('/')[-1]  # Get model name from path

                # Perform inference for the current sentence
                prediction = predict(model, tokenizer, [sentence])[0]  # Only one sentence
                f.write(f"Model: {model_name}\nOutput: {prediction}\n\n")
            f.write("=" * 50 + "\n\n")  # Separator between sentences

if __name__ == "__main__":
    main()
