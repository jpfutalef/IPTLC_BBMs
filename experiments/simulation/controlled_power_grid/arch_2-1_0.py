"""
GBM architecture with Power Grid using BBM PF

                0
                |
ARCHITECTURE:   0
                |
                1
Author: Juan-Pablo Futalef
"""

import time
import warnings
import numpy as np
from pathlib import Path
import dill as pickle
import torch
from importlib import reload
import pandas as pd
import tqdm

import greyboxmodels.cpsmodels.Input as Input
import greyboxmodels.cpsmodels.cyberphysical.ControlledPowerGrid.cases as cpg_cases
from greyboxmodels.scenariogeneration.MonteCarlo import MonteCarlo
from greyboxmodels.bbmcpsmodels.cyber import feedforward_nn_opf as opf_bbm

warnings.filterwarnings('ignore')

# Simulation parameters
SIM_MISSION_TIME = 3600 * 24 * 4
SIM_STEP_TIME = 15 * 60
SIM_INITIAL_TIME = 0.
MAX_EXECUTION_TIME = 3600 * 8

# Load the case
PF_BBM = pf_bbm.BBM1_SimpleNet(52, 10)
PF_BBM.load_state_dict(torch.load("models/BBM1_SimpleNet_MinMaxNormalizedOPF_20240321-163701.pt"))

# OPF_BBM = opf_bbm.BBM2_DeepNN(52, 10)
# OPF_BBM.load_model("models\BBM2-deep_MinMaxNormalizedOPF_20240321-164224.pt")

# Get the name of the OPF BBM
OPF_BBM_NAME = OPF_BBM.__class__.__name__

# The path
SAVE_TO = f"data/monte_carlo/controlled_power_grid/arch_1-0_1/{OPF_BBM_NAME}/{time.strftime('%Y-%m-%d_%H-%M-%S')}"

# Set the plant
SIM_PLANT = cpg_cases.case14(cc_type="data-driven", opf_bbm=OPF_BBM)

#%% Get initial condition and stimuli from WBM simulations
WBM_simulation_folder = "D:/projects/CPS-SenarioGeneration/data/monte_carlo/controlled_power_grid/2024-03-20_18-55-20/"
WBM_simulation_folder = Path(WBM_simulation_folder)

# Print the WBM report
with open(WBM_simulation_folder / "report.pkl", "rb") as f:
    report = pickle.load(f)

# Get the number of simulations
#%% Iterate all files and obtain realizations of the external stimuli
print("Loading external stimuli realizations...")
locs = [x for x in WBM_simulation_folder.iterdir() if x.is_file() and x.suffix == ".pkl" and "simulation" in x.stem]
E_realizations = []
for file in tqdm.tqdm(locs):
    with open(file, "rb") as f:
        data = pickle.load(f)
        df = pd.DataFrame(data["uncontrolled_inputs"], index=data["time"])
        E_realizations.append(df)

NUMBER_OF_TRIALS = len(E_realizations)

#%% Create the realizations
reload(Input)
EXTERNAL_STIMULI = Input.Input(realizations=E_realizations)

#%% Open a single simulation to obtain the initial condition
target_file = WBM_simulation_folder / "simulation_0.pkl"
with open(target_file, "rb") as f:
    data = pickle.load(f)
    SIM_INITIAL_CONDITION = data["initial_state"]


#%% Run simulation
if __name__ == '__main__':
    MC = MonteCarlo.MonteCarlo(SIM_PLANT,
                               SIM_INITIAL_CONDITION,
                               SIM_MISSION_TIME,
                               SIM_STEP_TIME,
                               NUMBER_OF_TRIALS,
                               SAVE_TO,
                               MAX_EXECUTION_TIME,
                               external_stimuli=EXTERNAL_STIMULI,
                               t0=SIM_INITIAL_TIME,
                               verbose=True)

    MC.run()
