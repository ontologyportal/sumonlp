import time
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# Import your dataset class
from datamodules.dataset import L2LDataset  # Update the import path if needed

# Path to your dataset file
data_path = "/home/angelos.toutsios.gr/data/Thesis_dev/L2L_model_training/data/sumo_dataset_500k_suffled_20250217_200924.json"

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
