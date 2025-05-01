import torch
from transformers import AutoTokenizer
from pathlib import Path
import sys
import json

# Calculate the project root (one level up from the src directory)
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.modules.T5model import L2LModel

prefix = "Convert the following sentence to SUO-KIF: "

# Path to the trained checkpoint
# CHECKPOINT_PATH = "/home/angelos.toutsios.gr/data/Thesis_dev/L2L_model_training/out/2025-03-06_09-12-34-good/lightning_logs/version_0/checkpoints/epoch=8-val_loss=0.00493.ckpt" # Load model from checkpoint
CHECKPOINT_PATH = sys.argv[1]
if not CHECKPOINT_PATH.lower().endswith(".ckpt"):
  print("Invalid checkpoint path. Please provide a .ckpt file.")
  sys.exit(1)

print(f"Loading model from {CHECKPOINT_PATH}...")
model = L2LModel.load_from_checkpoint(CHECKPOINT_PATH)
model.eval()  # Set model to evaluation mode
model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer (same as the one used for training)
tokenizer = AutoTokenizer.from_pretrained(model.hparams["model_name"])

def generate_output(input_text):
    """Generate logical output for a given input sentence."""
    print(f"Input Text: {repr(input_text)}")  # Debug input format

    # inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    inputs = tokenizer(input_text, padding=True, truncation=True, max_length=256, return_tensors="pt").to(model.device)
    # print(f"Tokenized Input: {inputs}")  # Debug tokenized format

    model.model.eval()

    with torch.no_grad():
        # output_ids = model.model.generate(**inputs, max_length=256)
        output_ids = model.model.generate(
        **inputs,
        max_length=256,
        # min_length=30,  # Force longer outputs
        # repetition_penalty=2.0,  # Penalize repetitive short outputs
        # do_sample=True,  # Enable sampling instead of greedy decoding
        # temperature=0.7,  # Introduce randomness
        # top_k=50,  # Limit sampling to top 50 tokens
        # top_p=0.9  # Nucleus sampling
        )



    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    # print("Generated Output:", repr(generated_text))  # To debug whitespace issues
    return generated_text

    # return tokenizer.decode(output_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    print("L2L Model is ready! Type your input (or 'exit' to quit).")

    # print(generate_output("Translate this sentence into logic: The cat is on the mat."))
    # print(generate_output("Explain the meaning of life."))
    # print(generate_output("Logic test: If A is taller than B and B is taller than C, who is the tallest?"))


    while True:
        user_input = input("\nEnter input: ").strip()
        if user_input.lower() == "exit":
            print("Exiting...")
            break

        output = generate_output(prefix + user_input)
        print(f"Output: {output}")