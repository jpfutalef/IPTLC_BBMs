"""
A script to create the specific Input-Output dataset for the TLCN from tabular data obtained from simulations.

Author: Juan-Pablo Futalef
"""

# Imports
import os
from pathlib import Path
import dill as pickle
import numpy as np
import time
import torch
import tqdm
import sys
import pandas as pd

from greyboxmodels.modelbuild import Plant

#%% Project roots
scenarios_root = Path(os.environ['SCENARIOGENERATION_ROOT'])
bbm_root = Path(os.environ['BBM_ROOT'])
wbm_root = Path(os.environ['MODELS_ROOT'])

#%% Working data folders
origin_folder = scenarios_root / "data/iptlc/RESTART/if1_dynamic_network_level_vulnerability/2024-05-09_15-09-08/tabular_scenarios/raw_sim_data/"
destination_folder = bbm_root / "data/IO-datasets/TLCN/RESTART/"

# Create the destination folder
os.makedirs(destination_folder, exist_ok=True)

#%% Open plant
plant_file = wbm_root / "data/wbm-models/iptlc_ieee14-deterministic_tlcn7_wbm.pkl"
plant = Plant.load(plant_file)

# Get the indices of the columns
idx_tlcn_bottom_up = plant.state_idx.tlc_network_bottom_up
idx_tlcn_top_down = plant.state_idx.tlc_network_top_down

# %% Since there is a bottom-up and top-down transmission, we will have two sets of inputs and outputs.
destination_bottom_up = destination_folder / "bottom_up"
destination_top_down = destination_folder / "top_down"

# Files for the inputs
input_bottom_up_folder = destination_bottom_up / "input_arrays"
input_bottom_up_folder.mkdir(parents=True, exist_ok=True)

input_top_down_folder = destination_top_down / "inputs_arrays"
input_top_down_folder.mkdir(parents=True, exist_ok=True)

# Files for the outputs
output_bottom_up_folder = destination_bottom_up / "output_arrays"
output_bottom_up_folder.mkdir(parents=True, exist_ok=True)

output_top_down_folder = destination_top_down / "output_arrays"
output_top_down_folder.mkdir(parents=True, exist_ok=True)

#%% get the files to process
cond = lambda x: x.suffix == ".pkl" and not any(y in x.stem for y in ["_temp", "failure"])
files = [file for file in origin_folder.iterdir() if cond(file)]

#%% Develop the IO datasets
for file in tqdm.tqdm(files, file=sys.stdout):
    # Load the data
    with open(file, "rb") as f:
        sim_data = pickle.load(f)

    # Get the time of the simulation
    sim_time = sim_data["time"]

    # Get the state and step data
    state_data = np.array(sim_data["state"])
    step_data = sim_data["step_data"]

    # Create the input dataframe
    index = pd.Index(sim_time, name="time")
    input_columns = Plant.get_variables_names(plant.tlc_network.state_idx)

    X_bottom_up = state_data[:, idx_tlcn_bottom_up]
    X_top_down = state_data[:, idx_tlcn_top_down]

    df_X_bottom_up = pd.DataFrame(X_bottom_up, index=index, columns=input_columns)
    df_X_top_down = pd.DataFrame(X_top_down, index=index, columns=input_columns)

    # Get the targets
    target_data_bottom_up = {}
    target_data_top_down = {}
    for dk in step_data:
        # Get the results
        result_bottom_up = dk["bottom_up_transmission_result"]
        result_top_down = dk["top_down_transmission_result"]

        # Get the target data
        for pair, result_dict in result_bottom_up.items():
            # pair = str(pair)
            if pair not in target_data_bottom_up:
                target_data_bottom_up[pair] = []
            target_data_bottom_up[pair].append(int(result_dict["success"]))

        for pair, result_dict in result_top_down.items():
            # pair = str(pair)
            if pair not in target_data_top_down:
                target_data_top_down[pair] = []
            target_data_top_down[pair].append(int(result_dict["success"]))

    # Turn into dataframes, making the columns a multiindex
    bottom_up_columns = pd.MultiIndex.from_tuples(target_data_bottom_up.keys(), names=["from", "to"])
    top_down_columns = pd.MultiIndex.from_tuples(target_data_top_down.keys(), names=["from", "to"])

    df_Y_bottom_up = pd.DataFrame.from_dict(target_data_bottom_up)
    df_Y_bottom_up.columns = bottom_up_columns
    df_Y_bottom_up.index = index

    df_Y_top_down = pd.DataFrame(target_data_top_down)
    df_Y_top_down.columns = top_down_columns
    df_Y_top_down.index = index

    # Save inputs
    intput_bottom_up_file = input_bottom_up_folder / f"{file.stem}.csv"
    input_top_down_file = input_top_down_folder / f"{file.stem}.csv"

    df_X_bottom_up.to_csv(intput_bottom_up_file)
    df_X_top_down.to_csv(input_top_down_file)

    # Save the output
    output_bottom_up_file = output_bottom_up_folder / f"{file.stem}.csv"
    output_top_down_file = output_top_down_folder / f"{file.stem}.csv"

    df_Y_bottom_up.to_csv(output_bottom_up_file)
    df_Y_top_down.to_csv(output_top_down_file)

