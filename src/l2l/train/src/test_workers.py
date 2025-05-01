import time
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import os

# Import your dataset class
from datamodules.dataset import L2LDataset  # Update the import path if needed

# Path to your dataset file
${SUMO_NLP_HOME}/src/l2l/train/data/test10k.json

sumo_nlp_home = os.environ.get('SUMO_NLP_HOME')
l2l_home = os.path.join(sumo_nlp_home, 'src', 'l2l')
# File paths
data_path = os.path.join(l2l_home, 'train', 'data', 'input_l2l.json')

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

# Initialize dataset
dataset = L2LDataset(data_path, tokenizer)

# Function to test different num_workers values
def find_optimal_workers(dataset, batch_size=32):
    for num_workers in [8, 4, 2, 0]:
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        start = time.time()
        for _ in range(10):  # Load 10 batches
            _ = next(iter(dataloader))
        end = time.time()
        print(f"num_workers={num_workers}: {end - start:.2f} sec")

# Run test
find_optimal_workers(dataset)
