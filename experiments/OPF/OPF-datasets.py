"""
A script to develop the datasets to train a data-driven model (surrogate model) for the Optimal Power Flow (OPF) problem.

The goal is to generate the inputs and outputs for the OPF problem, normalize them, and save them to files.

Author: Juan-Pablo Futalef
"""

import os
from pathlib import Path
import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

import greyboxmodels.cpsmodels.physical.electrical.PowerFlowPowerGrid as PG
import greyboxmodels.cpsmodels.Plant as Plant

# Set the working directory
print(f"Current working directory: {os.getcwd()}")

# %% Folder containing the data
# source_folder = Path("D:/projects/Hierarchical_CPS_models/data/simulations/controlled_exponential_pg/20240311_011412/")
source_folder = Path("D:/projects/CPS-SenarioGeneration/data/monte_carlo/controlled_power_grid/2024-03-20_18-55-20")
destination_folder = Path("./data/IO-datasets/OPF/", source_folder.name)

os.makedirs(destination_folder, exist_ok=True)

print(f"Source folder: {source_folder.resolve()}")
print(f"[Created] Destination folder: {destination_folder.resolve()}")

# %% Functions
def get_opf_data(filepath: Path):
    # Create a function that receives a path to a simulation and returns the inputs and outputs
    # Open the data
    with open(filepath, "rb") as f:
        sim_data = pickle.load(f)

    # Get the inputs and outputs
    opf_inputs = np.array([x['control_center_step_data']["opf_input"] for x in sim_data['step_data']])
    opf_outputs = np.array([x['control_center_step_data']["opf_output"] for x in sim_data['step_data']])
    output_names = Plant.get_variables_names(sim_data["plant"].control_center.state_idx)

    return opf_inputs, opf_outputs

def get_output_names(filepath: Path):
    # Open the data
    with open(filepath, "rb") as f:
        sim_data = pickle.load(f)

    output_names = Plant.get_variables_names(sim_data["plant"].control_center.state_idx)

    return output_names


# Now, a function that iterates over all the simulations and returns the inputs and outputs in a single numpy array
def get_opf_data_all(data_folder: Path):
    # Create an empty list to store the inputs and outputs
    inputs = []
    outputs = []

    # Create a list of the target folders: they are called "simulation_0.pkl", "simulation_1.pkl", etc.
    target_folders = [f for f in data_folder.iterdir() if
                      f.is_file() and f.name.startswith("simulation") and f.name.endswith(".pkl")]

    # Iterate over all the simulations and get the inputs and outputs for each one
    for f in tqdm.tqdm(target_folders):
        # Get the inputs and outputs
        opf_inputs, opf_outputs = get_opf_data(f)

        # Append the inputs and outputs to the lists
        inputs.append(opf_inputs)
        outputs.append(opf_outputs)

    # Concatenate the inputs and outputs
    inputs_matrix = np.concatenate(inputs, axis=0)
    outputs_matrix = np.concatenate(outputs, axis=0)

    # Get the output names
    output_names = get_output_names(target_folders[0])

    # Get the plant
    with open(data_folder / "plant.pkl", "rb") as f:
        plant = pickle.load(f)

    return inputs_matrix, outputs_matrix, plant, output_names


# Create a function to normalize an array as above
def min_max_normalize(array: np.ndarray, min_array: np.ndarray = None, max_array: np.ndarray = None):
    if min_array is None:
        min_array = array.min(axis=0)
        max_array = array.max(axis=0)

        min_array[min_array == max_array] = min_array[min_array == max_array] - 1
        max_array[min_array == max_array] = max_array[min_array == max_array]

    array_normalized = (array - min_array) / (max_array - min_array)

    return array_normalized, min_array, max_array


# %% An example of a simulation
target_simulation = "simulation_0.pkl"
target_simulation = source_folder / target_simulation
print(f"Testing the file: {target_simulation}")

opf_inputs, opf_outputs = get_opf_data(target_simulation)
print(f"CC inputs shape: {opf_inputs.shape}")
print(f"CC outputs shape: {opf_outputs.shape}")

#%% Get the output names
opf_output_names = get_output_names(target_simulation)

# Save the output names
output_names_path = destination_folder / "output_names.pkl"
with open(output_names_path, "wb") as f:
    pickle.dump(opf_output_names, f)

#%% Plots of the example
# Inputs in subplots
fig, axs = plt.subplots(4, 13, figsize=(20, 10), sharex=True, constrained_layout=True)
axs = axs.flatten()

for i in range(52):
    axs[i].plot(opf_inputs[:, i])
    axs[i].set_title(f"Input {i}")

fig.show()

# Plot the outputs in subplots
fig, axs = plt.subplots(2, 5, figsize=(20, 10), sharex=True, constrained_layout=True)
axs = axs.flatten()

for i in range(10):
    axs[i].plot(opf_outputs[:, i])
    axs[i].set_title(f"{opf_output_names[i]}")

fig.show()

#%% Develop the datasets using all the simulations
opf_inputs, opf_outputs, plant, output_names = get_opf_data_all(source_folder)

print(f"CC inputs shape: {opf_inputs.shape}")
print(f"CC outputs shape: {opf_outputs.shape}")

#%% Save the inputs and outputs to numpy arrays
print(f"Saving the inputs and outputs to numpy arrays...")
print(f"    Destination folder: {destination_folder.resolve()}")

inputs_path = destination_folder / "input.npy"
outputs_path = destination_folder / "output.npy"
output_names_path = destination_folder / "output_names.pkl"

np.save(inputs_path, opf_inputs)
np.save(outputs_path, opf_outputs)

# Also, save the output names
with open(output_names_path, "wb") as f:
    pickle.dump(output_names, f)

# %% Normalize the inputs and outputs
opf_inputs_normalized, min_opf_input, max_opf_input = min_max_normalize(opf_inputs)
opf_outputs_normalized, min_opf_output, max_opf_output = min_max_normalize(opf_outputs)

#%% Save the normalized inputs and outputs to numpy arrays
print(f"Saving the normalized inputs and outputs to numpy arrays...")
print(f"    Destination folder: {destination_folder.resolve()}")
inputs_normalized_path = destination_folder / "input_normalized.npy"
outputs_normalized_path = destination_folder / "output_normalized.npy"

np.save(inputs_normalized_path, opf_inputs_normalized)
np.save(outputs_normalized_path, opf_outputs_normalized)

# Also, save the min and max values
min_max_values = {"input_min": min_opf_input,
                  "input_max": max_opf_input,
                  "min_output": min_opf_output,
                  "max_output": max_opf_output}

min_max_values = {"output_min": min_opf_input,
                  "output_max": max_opf_input}

min_max_values_path = destination_folder / "normalization_spec.pkl"
with open(min_max_values_path, "wb") as f:
    pickle.dump(min_max_values, f)

#%% Ground truth data
