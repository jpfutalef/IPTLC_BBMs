import numpy as np
import pandas as pd
from pathlib import Path
import os
import tqdm
import dill as pickle

# Get roots of corresponding projects
origin_root = Path(os.environ['SCENARIOGENERATION_ROOT'])
destination_root = Path(os.environ['BBM_ROOT'])

# Point to target origin
origin_dataset_folder = origin_root / "data/iptlc/RESTART/if1_dynamic_network_level_vulnerability/2024-05-09_15-09-08/tabular_scenarios"

# Specify the destination
destination_dataset_folder = destination_root / "data/IO-datasets/TLCN/RESTART/"
destination_dataset_folder.mkdir(parents=True, exist_ok=True)

#%% Get all the files that are CSV
files = [file for file in origin_dataset_folder.iterdir() if file.suffix == ".csv"]

#%% Open plant to get specific column indices
# plant_file = origin_dataset_folder / "plant.pkl"
plant_file = "/mnt/d/projects/Hierarchical_CPS_models/data/wbm-models/iptlc_ieee14-deterministic_tlcn7_wbm.pkl"
with open(plant_file, "rb") as f:
    plant = pickle.load(f)

#%% Get the indices of the columns
idx_tlcn_bottom_up = plant.state_idx.tlc_network_bottom_up
idx_tlcn_top_down = plant.state_idx.tlc_network_top_down

# %% Get OD pair
pairs = [e for e in plant.tlc_network.base_graph.edges]
pairs_df = pd.DataFrame(pairs, columns=["origin", "destination"])
pairs_df.to_csv(destination_dataset_folder / "pairs.csv")

#%%Iterate the folder and load the data
# Containers
X = []
y = []

for file in tqdm.tqdm(files):
    # Load the data
    try:
        df = pd.read_csv(file)

    except Exception as e:
        print(f"Error loading {file}.")
        continue

    # Get the target

    # Get the features
    X.append(df.drop(columns=["target"]).values)

#%% Concatenate the data
X = np.concatenate(X)
y = np.concatenate(y)

# Print the shapes
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Save the data
np.save(destination_dataset_folder / "X.npy", X)
