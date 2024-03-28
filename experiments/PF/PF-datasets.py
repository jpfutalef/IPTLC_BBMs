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
source_folder = Path("D:/projects/CPS-SenarioGeneration/data/monte_carlo/controlled_power_grid/2024-03-20_18-55-20")
destination_folder = Path("./data/IO-datasets/PF/", source_folder.name)

os.makedirs(destination_folder, exist_ok=True)

print(f"Source folder: {source_folder.resolve()}")
print(f"[Created] Destination folder: {destination_folder.resolve()}")


# %% Functions
def get_pf_data(filepath: Path):
    # Create a function that receives a path to a simulation and returns the inputs and outputs
    # Open the data
    with open(filepath, "rb") as f:
        sim_data = pickle.load(f)

    # Get the plant
    plant = sim_data["plant"]

    # Get the inputs and outputs
    inputs = []
    outputs = []
    for step_dict in sim_data['step_data']:
        # Input
        power_demands = step_dict['power_demands']
        opf_output = step_dict['pg_control_input']
        x_pg = step_dict['state_post_update'][plant.state_idx.power_grid]
        u_pg = plant.power_grid.get_pf_inputs(x_pg, power_demands, opf_output)
        X = np.concatenate(u_pg)

        # Output is the response
        Y = step_dict["pg_response"]

        inputs.append(X)
        outputs.append(Y)

    # Concatenate the inputs and outputs
    X = np.vstack(inputs)
    Y = np.vstack(outputs)

    return X, Y, sim_data


# Now, a function that iterates over all the simulations and returns the inputs and outputs in a single numpy array
def get_pf_data_all(data_folder: Path):
    # Create an empty list to store the inputs and outputs
    inputs = []
    outputs = []

    # Create a list of the target folders: they are called "simulation_0.pkl", "simulation_1.pkl", etc.
    target_folders = [f for f in data_folder.iterdir() if
                      f.is_file() and f.name.startswith("simulation") and f.name.endswith(".pkl")]

    # Iterate over all the simulations and get the inputs and outputs for each one
    for f in tqdm.tqdm(target_folders):
        # Get the inputs and outputs
        pf_inputs, pf_outputs, sim_data = get_pf_data(f)
        inputs.append(pf_inputs)
        outputs.append(pf_outputs)

    # Concatenate the inputs and outputs
    inputs_matrix = np.concatenate(inputs, axis=0)
    outputs_matrix = np.concatenate(outputs, axis=0)

    # Get the plant
    with open(data_folder / "plant.pkl", "rb") as f:
        plant = pickle.load(f)

    return inputs_matrix, outputs_matrix, plant


# Create a function to normalize an array as above
def min_max_normalize(array: np.ndarray, min_array: np.ndarray = None, max_array: np.ndarray = None):
    if min_array is None:
        min_array = array.min(axis=0)
        max_array = array.max(axis=0)

        min_array[min_array == max_array] = min_array[min_array == max_array] - 1
        max_array[min_array == max_array] = max_array[min_array == max_array]

    array_normalized = (array - min_array) / (max_array - min_array)

    return array_normalized, min_array, max_array


def get_output_names(filepath: Path):
    # Open the data
    with open(filepath, "rb") as f:
        sim_data = pickle.load(f)

    output_names = Plant.get_variables_names(sim_data["plant"].power_grid.state_idx)

    return output_names


# %% An example of a simulation
target_simulation = "simulation_0.pkl"
target_simulation = source_folder / target_simulation
print(f"Testing the file: {target_simulation}")

pf_inputs, pf_outputs, target_sim_data = get_pf_data(target_simulation)
print(f"PF inputs shape: {pf_inputs.shape}")
print(f"PC outputs shape: {pf_outputs.shape}")

# %% Plots of the example
# Plot the inputs in subplots
# num_inputs = pf_inputs.shape[1]
# num_rows = np.sqrt(num_inputs).astype(int)
# num_cols = np.ceil(num_inputs / num_rows).astype(int)
# fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10), sharex=True, constrained_layout=True)
# axs = axs.flatten()
# for i in range(num_rows * num_cols):
#     if i >= num_inputs:
#         axs[i].axis("off")
#         continue
#     axs[i].plot(pf_inputs[:, i])
#     axs[i].set_title(f"Input {i}")
# fig.show()
#
# # Plot the outputs in subplots
# num_outputs = pf_outputs.shape[1]
# num_rows = np.sqrt(num_outputs).astype(int)
# num_cols = np.ceil(num_outputs / num_rows).astype(int)
# fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10), sharex=True, constrained_layout=True)
# axs = axs.flatten()
# for i in range(num_rows * num_cols):
#     if i >= num_outputs:
#         axs[i].axis("off")
#         continue
#     axs[i].plot(pf_outputs[:, i])
#     axs[i].set_title(f"Output {i}")
# fig.show()

# %% Develop the datasets using all the simulations
pf_inputs, pf_outputs, plant = get_pf_data_all(source_folder)

print(f"CC inputs shape: {pf_inputs.shape}")
print(f"CC outputs shape: {pf_outputs.shape}")

# %% Save the inputs and outputs to numpy arrays
print(f"Saving the inputs and outputs to numpy arrays...")
print(f"    Destination folder: {destination_folder.resolve()}")

inputs_path = destination_folder / "input.npy"
outputs_path = destination_folder / "output.npy"

np.save(inputs_path, pf_inputs)
np.save(outputs_path, pf_outputs)

# %% Normalize the inputs and outputs
inputs_normalized, min_input, max_input = min_max_normalize(pf_inputs)
outputs_normalized, min_output, max_output = min_max_normalize(pf_outputs)

# %% Save the normalized inputs and outputs to numpy arrays
print(f"Saving the normalized inputs and outputs to numpy arrays...")
print(f"    Destination folder: {destination_folder.resolve()}")
inputs_normalized_path = destination_folder / "input_normalized.npy"
outputs_normalized_path = destination_folder / "output_normalized.npy"

np.save(inputs_normalized_path, inputs_normalized)
np.save(outputs_normalized_path, outputs_normalized)

# Also, save the min and max values
min_max_values = {"min_input": min_input,
                  "max_input": max_input,
                  "min_output": min_output,
                  "max_output": max_output}

min_max_values_path = destination_folder / "normalization_spec.pkl"
with open(min_max_values_path, "wb") as f:
    pickle.dump(min_max_values, f)

# %% Ground truth data
