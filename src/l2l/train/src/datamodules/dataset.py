from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl
import torch
from torch.nn.utils.rnn import pad_sequence
import os
import jsonlines
import json

class L2LDataset(Dataset):

    def __init__(self, data_path, input_filename):

        # input_filename = os.path.basename(input_file_path)
        # input_filename_no_ext = os.path.splitext(input_filename)[0]
        self.tokenized_output_file = os.path.join(data_path, input_filename+".jsonl")
        self.index_file = os.path.join(data_path, input_filename + ".index")


        print(f"Loading pre-tokenized dataset from: {self.tokenized_output_file}")

        # Load precomputed line positions (byte offsets)
        with open(self.index_file, "r") as f:
            self.offsets = json.load(f)

        self.dataset_size = len(self.offsets)
        print(f"Total samples in dataset: {self.dataset_size}")

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        with open(self.tokenized_output_file, "r") as f:
            f.seek(self.offsets[idx])  # 🚀 Jump to the exact line in O(1) time
            line = f.readline()
            item = json.loads(line.strip())

            input_ids = torch.as_tensor(item['input_ids']).clone().detach().squeeze(0)  # Shape should now be (N,)
            attention_mask = torch.as_tensor(item['attention_mask']).clone().detach().squeeze(0)
            output_ids = torch.as_tensor(item['output_ids']).clone().detach().squeeze(0)

            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': output_ids,
            }


    def collate_fn(batch):
        input_ids = []
        attention_mask = []
        output_ids = []

        for item in batch:
            # Append the tensors directly to the lists
            input_ids.append(torch.as_tensor(item['input_ids']).clone().detach())
            attention_mask.append(torch.as_tensor(item['attention_mask']).clone().detach())
            output_ids.append(torch.as_tensor(item['labels']).clone().detach())

        # Pad the sequences to the maximum length in each batch
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=0)

        # print('-----------DONE-----------')
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': output_ids,
        }


class L2LDataModule(pl.LightningDataModule):
    def __init__(self, **config):
        super().__init__()

        self.data_path = config["data_dir"]  # Load pre-tokenized data
        self.data_name = config["data_name"]

        # self.input_file = config["input_file"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

    def setup(self, stage=None):
        dataset = L2LDataset(self.data_path, self.data_name)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=L2LDataset.collate_fn,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=1,
            persistent_workers=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=L2LDataset.collate_fn,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
        )