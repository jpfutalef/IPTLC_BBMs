"""
This scripts creates the input-output datasets for training the classification-based BBMs.


"""


from pathlib import Path
import dill as pickle
import os
import tqdm
import sys
import pandas as pd

from experiments.cpg.ra2 import bbm_root
from greyboxmodels.scenariogeneration.scenario_development import *

#%% Specify the scenarios' location
scenario_root = Path(os.environ.get("SCENARIOGENERATION_ROOT"))
bbm_root = Path(os.environ.get("BBM_ROOT"))

origin_folder = scenario_root / "data/iptlc/RESTART/if1_dynamic_network_level_vulnerability/2024-05-09_15-09-08/"


#%% INPUTS
















#%% LEGACY



origin_folder = scenario_root / Path(
    "../../../../../CPS-SenarioGeneration/data/iptlc/RESTART/if1_dynamic_network_level_vulnerability/2024-05-09_15-09-08/tabular_scenarios/raw_sim_data/")
bottom_up_destination = origin_folder.parent / "bottom_up_transmissions/"
top_down_destination = origin_folder.parent / "top_down_transmissions/"

#%% Create the destination folders
bottom_up_destination.mkdir(parents=True, exist_ok=True)
top_down_destination.mkdir(parents=True, exist_ok=True)

#%% get list of files
files = [file for file in origin_folder.iterdir() if file.suffix == ".pkl"]
bar = tqdm.tqdm(files, file=sys.stdout, leave=False)

#%% Iterate the files
top_down = {}
bottom_up = {}

for file in bar:
    # update the bar description
    bar.set_description(f"Processing {file.stem}")

    try:
        with open(file, "rb") as f:
            scenario = pickle.load(f)

            # prepare the transmission data
            transmission_top_down = []
            transmission_bottom_up = []

            for step_data in scenario['step_data']:
                # Get the transmission data
                transmission_top_down.append(step_data['top_down_transmission_result'])
                transmission_bottom_up.append(step_data['bottom_up_transmission_result'])

            # Save the transmission data
            top_down[file.stem] = transmission_top_down
            bottom_up[file.stem] = transmission_bottom_up

            # Save the dictionary in this status
            print(f"Saving {file.stem}.")
            with open(top_down_destination / f"{file.stem}.pkl", "wb") as f:
                pickle.dump(transmission_top_down, f)

            with open(bottom_up_destination / f"{file.stem}.pkl", "wb") as f:
                pickle.dump(transmission_bottom_up, f)

    except Exception as e:
        print(f"!! Error loading {file}.")
        continue

# %% Get processed files
processed_files = list(bottom_up.keys())

# %% Get the pairs from the first values in the dictionaries
pairs_bottom_up = list(bottom_up[processed_files[0]][0].keys())
pairs_top_down = list(top_down[processed_files[0]][0].keys())

#%% Transmissions into tabular data. Columns are the pairs, rows are the transmission results

# Iterate the files
for file in tqdm.tqdm(processed_files):
    data_bottom_up = bottom_up[file]
    data_top_down = top_down[file]

    # A container for the tabular data
    bottom_up_tabular = {pair: [] for pair in pairs_bottom_up}
    top_down_tabular = {pair: [] for pair in pairs_top_down}

    for dict_bottom_up, dict_top_down in zip(data_bottom_up, data_top_down):
        for pair, value in dict_bottom_up.items():
            bottom_up_tabular[pair].append(int(value["success"]))

        for pair, value in dict_top_down.items():
            top_down_tabular[pair].append(int(value["success"]))

    # Dataframes
    df_bottom_up = pd.DataFrame(bottom_up_tabular)
    df_top_down = pd.DataFrame(top_down_tabular)

    # Save the dataframes
    df_bottom_up.to_csv(bottom_up_destination / f"{file}_transmission_result.csv")
    df_top_down.to_csv(top_down_destination / f"{file}_transmission_result.csv")
