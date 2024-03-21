#!/usr/bin/env python
# coding: utf-8

# # Some predictions and plots

# In[1]:


import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import sys
from importlib import reload

import greyboxmodels.bbmcpsmodels.nn_creator as creator
import greyboxmodels.bbmcpsmodels.cyber.feedforward_nn_opf as opf_bbm

# Set the working directory
os.chdir("//")
print(f"Working directory: {os.getcwd()}")


# In[2]:


# Check GPU availability
device = creator.get_device()
print(device)


# In[3]:


# Open the gt data
input_data = np.load("data/OPF/20240226_184851/gt_inputs_minmax_normalized.npy")
output_data = np.load("data/OPF/20240226_184851/gt_outputs_minmax_normalized.npy")

# tensorize the data
input_data = torch.from_numpy(input_data).float().to(device)
output_data = torch.from_numpy(output_data).float().to(device)

# Shape of the data
input_size = input_data.shape[1]
output_size = output_data.shape[1]
print(f"Input data shape: {input_size}")
print(f"Output data shape: {output_size}")


# In[5]:


# Load the models
bbm1 = opf_bbm.BBM1_SimpleNet(input_size, output_size)
bbm1.load_state_dict(torch.load("models\BBM1_SimpleNet_MinMaxNormalizedOPF_20240226-200456_best.pt"))
bbm1.to(device)
bbm1.eval()

bbm2 = opf_bbm.BBM2_DeepNN(input_size, output_size)
bbm2.load_state_dict(torch.load("models\BBM2-deep_MinMaxNormalizedOPF_20240229-103342.pt"))
bbm2.to(device)
bbm2.eval()


# In[9]:


# Predictions
pred_inputs = bbm1(input_data)
pred_inputs2 = bbm2(input_data)


# In[10]:


# Font: Arial, size 12
plt.rcParams["font.family"] = "Arial"


# In[30]:


# Plot the predictions versus the ground truth
# Obtain the size of the output. THen, create an equal number of subplots and plot predictions versus ground truth in each subplot
output_size = output_data.shape[1]

# time
time = np.arange(0, len(output_data)) / 4

# Output names
output_names = ["Voltage angle 0",
                "Voltage angle 1",
                "Active Power Generation 0",
                "Active Power Generation 1",
                "Voltage magnitude 0",
                "Voltage magnitude 1",
                "Voltage magnitude SC 0",
                "Voltage magnitude SC 1",
                "Voltage magnitude SC 2"
                ]

# Calculate a nice number of rows and columns for the subplots
figsize = (12.8, 6.75)
rows = int(np.sqrt(output_size))
cols = int(np.ceil(output_size / rows))

# Create the figure
fig, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True, dpi=600)
axs = axs.flatten()

# Plot the predictions versus the ground truth
for i in range(rows * cols):
    # Delete the axis if there is no data to plot
    if i >= output_size:
        axs[i // cols, i % cols].axis("off")
        continue

    # Plot the prediction versus the ground truth
    axs[i].plot(time, output_data[:, i].cpu().detach().numpy(), label="Ground truth", color="black", linewidth=1)
    axs[i].plot(time, pred_inputs[:, i].cpu().detach().numpy(), label="Prediction", color="red", alpha=0.7, linewidth=1)#, linestyle="--")
    axs[i].plot(time, pred_inputs2[:, i].cpu().detach().numpy(), label="Prediction 2", color="blue", alpha=0.7, linewidth=1)#, linestyle=":")
    axs[i].set_title(f"{output_names[i]}")

    axs[i].set_xlim(0, 24 * 4)

    # y ticks
    axs[i].set_ylim(0, 1.1)
    axs[i].set_yticks([0, 0.25, 0.5, 0.75, 1])

    # If last row, add x label
    if i >= (rows - 1) * cols:
        axs[i].set_xlabel("Time (h)")

# Add figure title
fig.suptitle("OPF BBM predictions vs ground truth", fontsize=16)


# Add a legend at the bottom
_ = fig.legend(["Ground truth", "Prediction", "Prediction 2"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))


# In[29]:


# Specify the paths
datasets_folder = Path("data/OPF/2023-12-06_18-00-46")

# Let's open single simulation
single_input_normalized = np.load(datasets_folder / Path("OPF-NI1-O3/single_input.npy"))
single_input_raw = np.load(datasets_folder / Path("OPF-I1-O3/single_input.npy"))
single_output = np.load(datasets_folder / Path("OPF-I1-O3/single_output.npy"))

# Convert to torch tensors
single_input_normalized = torch.from_numpy(single_input_normalized).float().to(device)
single_input_raw = torch.from_numpy(single_input_raw).float().to(device)
single_output = torch.from_numpy(single_output).float().to(device)


# In[31]:


# Size of the input and output
input_size = single_input_normalized.shape[1]
output_size = single_output.shape[1]

print(f"Input size: {input_size}")
print(f"Output size: {output_size}")


# In[37]:


# Load the single example dataframes to extract the input/output names from the columns
input_df = pd.read_csv(datasets_folder / Path("OPF-I1-O3/single_input.csv"), index_col=0)
output_df = pd.read_csv(datasets_folder / Path("OPF-I1-O3/single_output.csv"), index_col=0)

# Get the names of the inputs and outputs
input_names = input_df.columns.values.tolist()
output_names = output_df.columns.values.tolist()


# In[14]:


# Load the models trained with the normalized dataset
OPF_BBM_1_normalized = opf_models.Net(input_size, output_size)
OPF_BBM_1_normalized.load_state_dict(torch.load("models/BBM1-2layers_OPF-NI1-O3_20240201-110613.pt"))
OPF_BBM_1_normalized.to(device)
OPF_BBM_1_normalized.eval()

OPF_BBM_2_normalized = opf_models.DeepNN(input_size, output_size)
OPF_BBM_2_normalized.load_state_dict(torch.load("models/BBM2-deep_OPF-NI1-O3_20240201-112030_best.pt"))
OPF_BBM_2_normalized.to(device)
OPF_BBM_2_normalized.eval()

# Load the models trained with the raw dataset
OPF_BBM_1_raw = opf_models.Net(input_size, output_size)
OPF_BBM_1_raw.load_state_dict(torch.load("models/BBM1-2layers_OPF-I1-O3_20240201-111317.pt"))
OPF_BBM_1_raw.to(device)
OPF_BBM_1_raw.eval()


# In[25]:


## Predictions using BBM1 with raw and normalized data
# Predictions using the normalized dataset and BBM1
prediction_normalized_1 = OPF_BBM_1_normalized(single_input_normalized)

# Predictions using the raw dataset and BBM1
prediction_raw_1 = OPF_BBM_1_raw(single_input_raw)
# Plot the predictions versus the ground truth
# Obtain the size of the output. THen, create an equal number of subplots and plot predictions versus ground truth in each subplot
print(f"Output size: {output_size}")

# Calculate a nice number of rows and columns for the subplots
figsize = (20, 10)
rows = int(np.sqrt(output_size))
cols = int(np.ceil(output_size / rows))

# Create the figure
fig, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)

try:
    y_limits
except NameError:
    y_limits = None

# Plot the predictions versus the ground truth
for i in range(rows * cols):
    # Delete the axis if there is no data to plot
    if i >= output_size:
        axs[i // cols, i % cols].axis("off")
        continue

    # Calculate the row and column of the subplot
    row = i // cols
    col = i % cols

    # Plot the prediction versus the ground truth
    axs[row, col].plot(single_output[:, i].cpu().detach().numpy(), label="Ground truth", color="black")
    axs[row, col].plot(prediction_raw_1[:, i].cpu().detach().numpy(), label="Prediction (raw input)", color="black",
                       linestyle="--")
    axs[row, col].plot(prediction_normalized_1[:, i].cpu().detach().numpy(), label="Prediction (normalized input)",
                       color="black", linestyle=":")
    axs[row, col].set_title(output_names[i])

    # Set the x limit
    axs[row, col].set_xlim([0, 96])

    # Set the y limit
    if y_limits is not None:
        # get the current y limits
        this_y_bot, this_y_up = axs[row, col].get_ylim()
        gen_y_bot, gen_y_up = y_limits[i]

        # check both limits: if bottom limit is lower than the current one, update it and viceversa
        if this_y_bot < gen_y_bot:
            y_limits[i][0] = this_y_bot
        if this_y_up > gen_y_up:
            y_limits[i][1] = this_y_up

        # set the limits
        axs[row, col].set_ylim(y_limits[i])

# Add a legend at the top
_ = fig.legend(["Ground truth", "Prediction (raw input)", "Prediction (normalized input)"], loc="upper center", ncol=3,bbox_to_anchor=(0.5, 1.05))


# In[27]:


# Predictions using the normalized dataset and BBM1
prediction_normalized_2 = OPF_BBM_2_normalized(single_input_normalized)


# In[28]:


# Plot the predictions versus the ground truth
# Obtain the size of the output. THen, create an equal number of subplots and plot predictions versus ground truth in each subplot
output_size = single_output.shape[1]
print(f"Output size: {output_size}")

# Calculate a nice number of rows and columns for the subplots
figsize = (20, 10)
rows = int(np.sqrt(output_size))
cols = int(np.ceil(output_size / rows))

# Create the figure
fig, axs = plt.subplots(rows, cols, figsize=figsize, constrained_layout=True)

# Plot the predictions versus the ground truth
for i in range(rows * cols):
    # Delete the axis if there is no data to plot
    if i >= output_size:
        axs[i // cols, i % cols].axis("off")
        continue

    # Calculate the row and column of the subplot
    row = i // cols
    col = i % cols

    # Plot the prediction versus the ground truth
    axs[row, col].plot(single_output[:, i].cpu().detach().numpy(), label="Ground truth", color="black")
    axs[row, col].plot(prediction_normalized_1[:, i].cpu().detach().numpy(), label="Prediction (BBM1)", color="black", linestyle=":")
    axs[row, col].plot(prediction_normalized_2[:, i].cpu().detach().numpy(), label="Prediction (BBM2)", color="red", linestyle="-.")
    axs[row, col].set_title(output_names[i])

    # Set the x limit
    axs[row, col].set_xlim([0, 96])

    # Set the y limit
    if y_limits is not None:
        # get the current y limits
        this_y_bot, this_y_up = axs[row, col].get_ylim()
        gen_y_bot, gen_y_up = y_limits[i]

        # check both limits: if bottom limit is lower than the current one, update it and viceversa
        if this_y_bot < gen_y_bot:
            y_limits[i][0] = this_y_bot
        if this_y_up > gen_y_up:
            y_limits[i][1] = this_y_up

        # set the limits
        axs[row, col].set_ylim(y_limits[i])

# Add a legend at the top
_ = fig.legend(["Ground truth", "Prediction (BBM1)", "Prediction (BBM2)"], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.05))


# In[ ]:




