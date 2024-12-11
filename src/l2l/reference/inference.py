import torch
import sys
import argparse
from transformers import AutoTokenizer, T5ForConditionalGeneration  # Use T5 classes
import warnings
import os

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)


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
    # parser = argparse.ArgumentParser(description='Run inference with the trained T5 model.')
    # parser.add_argument('--out_dir', required=True, help='Path to the model directory')
    # parser.add_argument('--inference_input_file', required=True, help='Input file for inference')
    # parser.add_argument('--inference_output_file', required=True, help='Output file for inference results')

    # args = parser.parse_args()

    # Load the model and tokenizer from the saved directory
    model = load_model(sys.argv[1])
    # tokenizer = T5Tokenizer.from_pretrained('./t5_model')
    tokenizer = AutoTokenizer.from_pretrained('t5-small')

    # Read input sentences from the input file
    with open('./input_l2l.txt', 'r') as f:
        sentences = [line.strip() for line in f.readlines()]

    # Perform inference
    predictions = predict(model, tokenizer, sentences)

    # Write outputs to the output file
    with open('./output_l2l.txt', 'w') as f:
        for sentence, prediction in zip(sentences, predictions):
            # f.write(f"Input: {sentence}\nOutput: {prediction}\n\n")
            f.write(f"{prediction}\n")

if __name__ == "__main__":
    main()
