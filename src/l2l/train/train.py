import json
from tokenize_data import tokenize_data
from dataset import CustomDataset, collate_fn  # Ensure this matches your actual class name
from model import train_model
from torch.utils.data import DataLoader
import time

# Paths to your data
input_file = 'data/input_sentences.txt'
output_file = 'data/output_logical.txt'
tokenized_output_file = 'data/tokenized_data.json'

# Step 1: Tokenize the data (uncomment if needed)
start_time = time.time()
# tokenize_data(input_file, output_file, tokenized_output_file)
print(f"Data tokenization completed in {time.time() - start_time:.2f} seconds.")

# Step 2: Load tokenized data
start_time = time.time()
try:
    with open(tokenized_output_file, 'r') as f:
        tokenized_data = json.load(f)
    print(f"Tokenized data loaded successfully in {time.time() - start_time:.2f} seconds.")
except FileNotFoundError:
    print(f"Error: The file {tokenized_output_file} was not found.")
    exit(1)

# Step 3: Create dataset
start_time = time.time()
dataset = CustomDataset(tokenized_data)  # Ensure this matches your actual class name
print(f"Dataset created successfully in {time.time() - start_time:.2f} seconds.")

# Step 4: Create DataLoader
start_time = time.time()
train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True) # One GPU
# train_loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn, shuffle=True, num_workers=8, pin_memory=True) # Multiple GPUs
print(f"DataLoader created successfully in {time.time() - start_time:.2f} seconds.")


# Step 5: Train the model
start_time = time.time()
model = train_model(train_loader)
print(f"Model trained successfully in {time.time() - start_time:.2f} seconds.")



