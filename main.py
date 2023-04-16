from typing import List

import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from create_vessel_train_set import load_dataset, split_datasets
from encoder import add_ordinal_label, VESSEL_ID, PREV_PORT, PORT


class PortPredictNeuralNetwork(nn.Module):
    def __init__(self, vessel_dim, port_dim, embedding_dim, hidden_dim, output_dim):
        super(PortPredictNeuralNetwork, self).__init__()
        self.vessel_embedding = nn.Embedding(vessel_dim, embedding_dim)
        self.port_embedding = nn.Embedding(port_dim, embedding_dim)        # This is our holy embedding layer

        # These are two fully connected layers
        self.layer_1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.layer_2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        # Pass input through the embedding layer
        vessel_embed = self.vessel_embedding(inputs[0])
        port_embed = self.port_embedding(inputs[1])
        embed = torch.cat((vessel_embed, port_embed), 1)

        # Pass input through the hidden layer
        out = F.relu(self.layer_1(embed))
        # Pass input through the final linear layer
        out = self.layer_2(out)
        out = F.log_softmax(out, 1)
        return out


def train_epoch(train_df, val_df):
    total_loss = 0
    accurate = 0
    for index, row in train_df[[VESSEL_ID, PREV_PORT, f"input_{VESSEL_ID}", f"input_{PREV_PORT}", "output_label"]].iterrows():
        # Every data instance is an input + label pair
        inputs = (torch.tensor(row[f"input_{VESSEL_ID}"], dtype=torch.long), torch.tensor(row[f"input_{PREV_PORT}"], dtype=torch.long))

        # Zero your gradients for every batch!
        model.zero_grad()

        # Run the forward pass, getting softmax probabilities
        output = model(inputs)
        predicted_ix = output.argmax().item()
        target = torch.tensor(row["output_label"], dtype=torch.long)
        if target.item() == predicted_ix:
            accurate += 1

        # Step 4. Compute your loss function
        loss = loss_function(output, target)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Get the Python number from a 1-element Tensor by calling tensor.item()
        total_loss += loss.item()

    val_err = run_validation(val_df)
    return total_loss, accurate / (train_df.shape[0]), val_err


def run_validation(val_df):
    vessel_input = torch.flatten(torch.tensor(np.array(val_df[f"input_{VESSEL_ID}"]).astype(int), dtype=torch.long))
    port_input = torch.flatten(torch.tensor(np.array(val_df[f"input_{PREV_PORT}"]).astype(int), dtype=torch.long))
    val_target = torch.flatten(torch.tensor(np.array(val_df["output_label"]).astype(int), dtype=torch.long))
    validation_output = model((vessel_input, port_input))
    val_loss = loss_function(validation_output, val_target)
    correct = torch.sum(torch.eq(val_target, validation_output.argmax(dim=1))).item()
    val_err = correct / val_df.shape[0]
    return val_err


def get_unique(columns: List[str], data_sets: List[pd.DataFrame]):
    values = []
    for column in columns:
        for data_set in data_sets:
            values.extend(list(data_set[column].unique()))
    return list(set(values))


dataset_df = load_dataset()
unique_vessels = get_unique([VESSEL_ID], [dataset_df])
unique_ports = get_unique([PORT, PREV_PORT], [dataset_df])

data_set_df, vessel_encoder = add_ordinal_label(unique_vessels, dataset_df, VESSEL_ID)
data_set_df, port_encoder = add_ordinal_label(unique_ports, data_set_df, PORT)
data_set_df, port_encoder = add_ordinal_label(unique_ports, data_set_df, PREV_PORT, outside_encoder=port_encoder)
data_set_df = data_set_df.rename(
    columns={f"{PORT}_ordinal": "output_label", f"{VESSEL_ID}_ordinal": f"input_{VESSEL_ID}", f"{PREV_PORT}_ordinal": f"input_{PREV_PORT}"}
)

train_set_df, val_set_df, test_set_df = split_datasets(data_set_df)

train_set_df = train_set_df.sample(frac=1).reset_index(drop=True)
val_set_df = val_set_df.sample(frac=1).reset_index(drop=True)
test_set_df = test_set_df.sample(frac=1).reset_index(drop=True)

# Initialize the model
vessels_size = len(unique_vessels)
ports_size = len(unique_ports)

hidden_layer_size = 64  # TODO: play with this
vessel_emb_dim = 1
port_emb_dim = 2
emb_output_dim = 2
model = PortPredictNeuralNetwork(vessels_size, ports_size, emb_output_dim, hidden_layer_size, ports_size)
optimizer = optim.SGD(model.parameters(), lr=0.005)

loss_function = nn.NLLLoss()

for epoch in range(100):
    curr_loss, train_accuracy, val_accuracy = train_epoch(train_set_df, val_set_df)
    print(f"Epoc: {epoch}; Loss: {round(curr_loss, 3)}; Train Accuracy: {round(train_accuracy*100, 2)}%; "
          f"Validation Accuracy: {round(val_accuracy*100, 2)}%")
# for epoch, loss, accuracy in losses:
#     print(f"Epoc: {epoch}; Loss: {round(loss, 3)}; Accuracy: {round(accuracy, 2)}")
