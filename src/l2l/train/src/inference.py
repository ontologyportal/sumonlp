import pandas as pd
from transformers import T5ForConditionalGeneration, AutoTokenizer
from pathlib import Path
import sys
import torch
import os
import warnings
import time

# Suppress logging warnings
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

# Calculate the project root (one level up from the src directory)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# from src.modules.T5model import L2LModel
from src.modules.T5model import L2LModel

sumo_nlp_home = os.environ.get('SUMO_NLP_HOME')
l2l_home = os.path.join(sumo_nlp_home, 'src', 'l2l')

def load_model(model_path):
    # Load the model from the specified path
    model = L2LModel.load_from_checkpoint(model_path)
    model.eval()  # Set the model to evaluation mode
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def read_input_file():
    with open(l2l_home+'/reference_new/input_l2l.txt', 'r') as f:
        sentences = [line.strip() for line in f.readlines()]
        return sentences


def predict(model, tokenizer, input_sentences):
    prefix = "Convert the following sentence to SUO-KIF: "
    predictions = []
    for sentence in input_sentences:
      if(not sentence.startswith('SentenceId:')):
        sentence = prefix + sentence
        # ✅ Prepare Inputs
        inputs = tokenizer(sentence,
                          return_tensors="pt",
                          padding=True,
                          truncation=True,
                          ).to(model.device)
        with torch.no_grad():
            output_ids = model.model.generate(**inputs, max_length=500)
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        predictions.append(outputs)
      else:
        predictions.append(sentence)
    return predictions

def save_predictions(predictions, output_file):
    with open(output_file, 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

def main():

    model_path = sys.argv[1]
    print(f"Loading model from {model_path}...")

    model = load_model(model_path)
    print("Model loaded successfully.")

    input_sentences = read_input_file()

    # Load tokenizer (same as the one used for training)
    tokenizer = AutoTokenizer.from_pretrained(model.hparams["model_name"])

    predictions = predict(model, tokenizer, input_sentences)

    output_file = l2l_home+'/reference_new/output_l2l.txt'
    save_predictions(predictions, output_file)
    print(f"Predictions saved to {output_file}.")


if __name__ == "__main__":
  start_time = time.time()
  main()
  print(f"Processing complete in {time.time() - start_time:.2f} seconds.")