import json

from sklearn.model_selection import train_test_split
import pandas as pd


def load_dataset():
    with open("dataset_10_samples.json", mode="r") as f:
        dataset_10 = json.load(f)
    dataset_df = pd.DataFrame(dataset_10)
    return dataset_df


def split_datasets(dataset_df):
    train_df, val_df = train_test_split(dataset_df, train_size=0.6, shuffle=False)
    val_df, test_df = train_test_split(val_df, test_size=0.5, shuffle=False)
    return train_df, val_df, test_df

# train_set_df, val_set_df, test_set_df = load_dataset()
