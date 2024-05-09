import os
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import torch
import sys
from importlib import reload

import greyboxmodels.bbmcpsmodels.nn_creator as creator
import greyboxmodels.bbmcpsmodels.cyber.feedforward_nn_opf as opf_bbm

# Set the working directory
print(f"Working directory: {os.getcwd()}")

# Check GPU availability
device = creator.get_device()
print(device)

#%% Open test sim_data
dataloaders_loc = Path("models/OPF/dataloaders_OPF_2024-03-20_18-55-20.pth").resolve()

print(f"Loading dataloaders from {dataloaders_loc}")
dataloaders = torch.load(dataloaders_loc)
print("     Dataloaders loaded successfully!")

test_dataset = dataloaders[2].dataset
input_data = test_dataset[:][0]
output_data = test_dataset[:][1]

if isinstance(input_data, np.ndarray):
    input_data = torch.from_numpy(input_data).float()
if isinstance(output_data, np.ndarray):
    output_data = torch.from_numpy(output_data).float()

# Move the sim_data to the device
input_data = input_data.to(device)
output_data = output_data.to(device)

# Shape of the sim_data
input_size = input_data.shape[1]
output_size = output_data.shape[1]
print(f"     Input sim_data shape: {input_size}")
print(f"     Output sim_data shape: {output_size}")
print(f"     Number of samples: {len(test_dataset)}")

#%% Get the output names
with open("sim_data/IO-datasets/OPF/2024-03-20_18-55-20/output_names.pkl", "rb") as f:
    output_names = pickle.load(f)

# %% Load the models
print("Loading models...")

bbm1 = opf_bbm.BBM1_SimpleNet(input_size, output_size)
bbm1.load_state_dict(torch.load("models/OPF/BBM1_SimpleNet_OPF_2024-03-20_18-55-20_20240326-143608_best.pt"))
bbm1.to(device)
bbm1.eval()

bbm2 = opf_bbm.BBM2_DeepNN(input_size, output_size)
bbm2.load_state_dict(torch.load("models/OPF/BBM2-deep_OPF_2024-03-20_18-55-20_20240326-143955_best.pt"))
bbm2.to(device)
bbm2.eval()

print("     Models loaded successfully!")

#%% Predictions
pred_bbm1 = bbm1(input_data)
pref_bbm2 = bbm2(input_data)

#%% 45 degree line for the predictions
line = np.linspace(0, 1, 100)

# Plot the predictions versus the test sim_data
num_outputs = output_data.shape[1]
n_cols = 5
n_rows = int(np.ceil(num_outputs / n_cols))
colors = matplotlib.colormaps["tab10"]

fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 5), sharex=True, sharey=True)
axs = axs.flatten()

for i in range(n_rows * n_cols):
    ax = axs[i]
    if i >= num_outputs:
        ax.axis("off")
        continue

    # Plot the 45 degree line
    ax.plot(line, line, color="black", linestyle="--", linewidth=0.6)

    # Plot the test sim_data versus the predictions
    ax.plot(output_data[:, i].cpu().detach().numpy(), pred_bbm1[:, i].cpu().detach().numpy(),
            'o', color=colors(0/10), alpha=0.3, linewidth=1, markersize=1)

    ax.plot(output_data[:, i].cpu().detach().numpy(), pref_bbm2[:, i].cpu().detach().numpy(),
            'o', color=colors(1/10), alpha=0.3, linewidth=1, markersize=1)

    ax.set_title(f"{output_names[i]}")
    # If first column, add y label
    if i % n_cols == 0:
        ax.set_ylabel("Prediction")

    # If last row, add x label and set the x lim
    if i >= (n_rows - 1) * n_cols:
        ax.set_xlabel("Real value")

    # Grid
    ax.grid(True, linestyle="--", linewidth=0.5)


# Legend at the bottom. Make the markers bigger and with alpha=1
fig.legend(["Reference", "BBM1", "BBM2"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0),
           markerscale=5, fontsize=8)

# Ensure the legend is not cut off
fig.tight_layout(rect=[0, 0.05, 1, 1])

fig.show()

#%%
# histograms of the ground truth outputs
fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 5), sharex=True, sharey=True)
axs = axs.flatten()
for i in range(n_rows * n_cols):
    ax = axs[i]
    if i >= num_outputs:
        ax.axis("off")
        continue

    ax.hist(output_data[:, i].cpu().detach().numpy(), bins=50, alpha=0.5)
    ax.grid(True, linestyle="--", linewidth=0.5)

#%%
test_dataset = dataloaders[2].dataset
input_data = test_dataset[:][0]
output_data = test_dataset[:][1]
fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)

ax.plot(output_data[:, 0].cpu().detach().numpy(), output_data[:, 1].cpu().detach().numpy(), "o", markersize=.5, alpha=0.3)
ax.set_xlabel("Gen 0")
ax.set_ylabel("Gen 1")
ax.set_title("Test sim_data")
fig.show()

#%%
training_dataset = dataloaders[0].dataset
input_data = training_dataset[:][0]
output_data = training_dataset[:][1]

fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)

ax.plot(output_data[:, 0].cpu().detach().numpy(), output_data[:, 1].cpu().detach().numpy(),
        "o", markersize=.5, alpha=0.3)
ax.set_xlabel("Gen 0")
ax.set_ylabel("Gen 1")
ax.set_title("Training sim_data")
fig.show()

#%%
training_dataset = dataloaders[0].dataset
input_data = training_dataset[:][0]
output_data = training_dataset[:][1]

fig, ax = plt.subplots(1, 1, figsize=(10, 5), sharex=True, sharey=True)

ax.plot(output_data[:, 7].cpu().detach().numpy(), output_data[:, 8].cpu().detach().numpy(),
        "o", markersize=.5, alpha=0.3)
ax.set_ylim(0.95, 1.02)
ax.set_xlim(0.9, 1.02)
ax.set_xlabel("Gen 0")
ax.set_ylabel("V0")
ax.set_title("Training sim_data")
fig.show()
