from typing import List

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from create_vessel_train_set import load_dataset, split_datasets, dataloader_from_df
from encoder import add_ordinal_label, VESSEL_ID, PREV_PORT, PORT

import matplotlib.pyplot as plt


class PortPredictNeuralNetwork(nn.Module):
    def __init__(self, vessel_dim, port_dim, embedding_dim, hidden_dim, output_dim):
        super(PortPredictNeuralNetwork, self).__init__()
        self.vessel_embedding = nn.Embedding(vessel_dim, embedding_dim)
        self.port_embedding = nn.Embedding(port_dim, embedding_dim)  # This is our holy embedding layer

        # These are two fully connected layers
        self.layer_1 = nn.Linear(embedding_dim * 2, hidden_dim)
        # self.layer_2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        # Pass input through the embedding layer
        vessel_embed = self.vessel_embedding(inputs[0])
        port_embed = self.port_embedding(inputs[1])
        embed = torch.cat((vessel_embed, port_embed), 2)

        # Pass input through the hidden layer
        out = F.relu(self.layer_1(embed))
        # out = F.relu(self.layer_2(out))
        # Pass input through the final linear layer
        out = self.layer_3(out)
        out = F.log_softmax(out, 2)
        return out


def train_epoch(train_dl: DataLoader, val_dl):
    running_loss = []
    loss_weights = []
    correct = 0
    for i, (inputs, target) in enumerate(train_dl):
        # Zero your gradients for every batch!
        model.zero_grad()

        # Run the  forward pass, getting softmax probabilities
        output = model(inputs)
        correct += torch.sum(output.argmax(2) == target).item()

        # Step 4. Compute your loss function
        loss = loss_function(output.squeeze(), target.squeeze())
        weight = inputs[0].size()[0]
        running_loss.append(loss.item() * weight)
        loss_weights.append(weight)

        # Step 5. Do the backward pass and update the gradient
        loss.backward()

        # Adjust learning weights
        optimizer.step()

    val_loss, val_acc = run_validation(val_dl)
    return sum(running_loss) / sum(loss_weights), 100 * correct / len(train_set_df), val_loss, val_acc


def run_validation(val_dl):
    running_loss = []
    loss_weights = []
    correct = 0
    for val_inputs, val_target in val_dl:
        val_output = model(val_inputs)
        val_loss = loss_function(val_output.squeeze(), val_target.squeeze())
        weight = val_inputs[0].size()[0]
        running_loss.append(val_loss.item() * weight)
        loss_weights.append(weight)
        correct += torch.sum(val_output.argmax(2) == val_target).item()
    return sum(running_loss) / sum(loss_weights), 100 * correct / len(val_set_df)


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

# -------------------------------- #

hidden_layer_size = 64

emb_output_dim = 3
learning_rate = 0.001
batch_size = 32

train_dataloader = dataloader_from_df(train_set_df, batch_size)
validation_dataloader = dataloader_from_df(val_set_df, 100)

# Initialize the model
vessels_size = len(unique_vessels)
ports_size = len(unique_ports)

model = PortPredictNeuralNetwork(vessels_size, ports_size, emb_output_dim, hidden_layer_size, ports_size)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

loss_function = nn.NLLLoss()

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

epochs = range(600)

loss_changed = False

for epoch in epochs:
    curr_loss, train_accuracy, curr_val_loss, val_accuracy = train_epoch(train_dataloader, validation_dataloader)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}; Loss: {round(curr_loss, 5)}; Train Accuracy: {round(train_accuracy, 2)}%; "
              f"Validation Loss: {round(curr_val_loss, 5)}; Validation Accuracy: {round(val_accuracy, 2)}%")
    # if curr_val_loss < 3.7 and not loss_changed:
    #     learning_rate = 0.005
    #     optimizer.param_groups[0]['lr'] = 0.001
    #     loss_changed = True
    train_losses.append(curr_loss)
    train_accuracies.append(train_accuracy)
    val_losses.append(curr_val_loss)
    val_accuracies.append(val_accuracy)

plt.figure()
plt.plot(list(map(lambda x: x + 1, epochs)), train_losses, list(map(lambda x: x + 1, epochs)), val_losses)
plt.title(
    f"Train vs. Validation Loss lr={learning_rate}, em_dim={emb_output_dim}, batch_size={batch_size}\ntrainset size {len(train_set_df)}, vessels {len(unique_vessels)}, ports {len(unique_ports)}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "validation"])
plt.show()
