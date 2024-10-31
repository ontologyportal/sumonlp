from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data

    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx):
        # item = self.tokenized_data[idx]
        # input_ids = torch.tensor(item['input_ids'])
        # attention_mask = torch.tensor(item['attention_mask'])
        # output_ids = torch.tensor(item['output_ids'])
        # output_attention_mask = torch.tensor(item['output_attention_mask'])

        item = self.tokenized_data[idx]
        input_ids = torch.tensor(item['input_ids']).squeeze(0)  # Shape should now be (N,)
        attention_mask = torch.tensor(item['attention_mask']).squeeze(0)
        output_ids = torch.tensor(item['output_ids']).squeeze(0)
        output_attention_mask = torch.tensor(item['output_attention_mask']).squeeze(0)


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'output_ids': output_ids,
            'output_attention_mask': output_attention_mask
        }


def collate_fn(batch):
    input_ids = []
    attention_mask = []
    output_ids = []
    output_attention_mask = []

    for item in batch:
        # Append the tensors directly to the lists
        input_ids.append(torch.tensor(item['input_ids']))
        attention_mask.append(torch.tensor(item['attention_mask']))
        output_ids.append(torch.tensor(item['output_ids']))
        output_attention_mask.append(torch.tensor(item['output_attention_mask']))

    # Debugging: Check shapes before padding
    # for i, tensor in enumerate(input_ids):
    #     print(f"Input IDs {i} shape: {tensor.shape}")

    # Pad the sequences to the maximum length in each batch
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    output_ids = torch.nn.utils.rnn.pad_sequence(output_ids, batch_first=True, padding_value=0)
    output_attention_mask = torch.nn.utils.rnn.pad_sequence(output_attention_mask, batch_first=True, padding_value=0)

    # print('-----------DONE-----------')
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'output_ids': output_ids,
        'output_attention_mask': output_attention_mask
    }