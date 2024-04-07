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
# source_folder = Path("D:/projects/CPS-SenarioGeneration/data/monte_carlo/controlled_power_grid/2024-03-20_18-55-20")
source_folder = Path("D:/projects/CPS-SenarioGeneration/data/cpg/MonteCarlo/2024-04-03_18-06-45")
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

        # Filter up to column 80 to the end TODO HARDCODED!!!!
        Y = Y[:80]

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
    inputs_matrix = np.vstack(inputs)
    outputs_matrix = np.vstack(outputs)

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
min_max_values = {"input_min": min_input,
                  "input_max": max_input,
                  "output_min": min_output,
                  "output_max": max_output,
                  "type": "min_max"}

min_max_values_path = destination_folder / "normalization_spec.pkl"
with open(min_max_values_path, "wb") as f:
    pickle.dump(min_max_values, f)

# %% Construct a dataset without the binary variables at the output
# Get the indices of the binary variables
binary_indices = plant.power_grid.state_idx.get_binary_indices()

# Filter
output_no_binary = np.delete(pf_outputs, binary_indices, axis=1)
output_no_binary_normalized = np.delete(outputs_normalized, binary_indices, axis=1)

# Save the dataset
output_no_binary_path = destination_folder / "output_no_binary.npy"
output_no_binary_normalized_path = destination_folder / "output_no_binary_normalized.npy"
