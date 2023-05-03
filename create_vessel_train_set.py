import json

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd

from encoder import VESSEL_ID, PREV_PORT


class VesselPortDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_vessel_id = self.data.iloc[index][f"input_{VESSEL_ID}"]
        input_port_id = self.data.iloc[index][f"input_{PREV_PORT}"]
        output_label = self.data.iloc[index]["output_label"]

        return (
            (torch.tensor(input_vessel_id, dtype=torch.long), torch.tensor(input_port_id, dtype=torch.long)),
            torch.tensor(output_label, dtype=torch.long)
        )


def dataloader_from_df(data: pd.DataFrame, batch_size: int):
    dataset = VesselPortDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def load_dataset():
    with open("dataset_10_samples.json", mode="r") as f:
        dataset_10 = json.load(f)
    with open("dataset_24_samples.json", mode="r") as f:
        dataset_24 = json.load(f)
    dataset_df = pd.DataFrame(dataset_10)
    return dataset_df


def split_datasets(dataset_df):
    train_df, val_df = train_test_split(dataset_df, train_size=0.6, shuffle=False)
    val_df, test_df = train_test_split(val_df, test_size=0.5, shuffle=False)
    return train_df, val_df, test_df

# train_set_df, val_set_df, test_set_df = load_dataset()
