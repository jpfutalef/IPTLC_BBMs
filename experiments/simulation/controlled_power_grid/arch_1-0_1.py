"""
GBM architecture with Control Center using BBM OPF

                1
                |
ARCHITECTURE:   0
                |
                0
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
OPF_BBM = opf_bbm.BBM1_SimpleNet(52, 10)
#OPF_BBM.load_state_dict(torch.load("models/BBM1_SimpleNet_MinMaxNormalizedOPF_20240321-163701.pt"))
OPF_BBM.load_state_dict(torch.load("models/OPF/BBM1_SimpleNet_OPF_2024-04-03_18-06-45_20240408-003640.pt"))

# OPF_BBM = opf_bbm.BBM2_DeepNN(52, 10)
# OPF_BBM.load_model("models\BBM2-deep_MinMaxNormalizedOPF_20240321-164224.pt")

# Get the name of the OPF BBM
OPF_BBM_NAME = OPF_BBM.__class__.__name__

# Get the normalization spec
with open("data/IO-datasets/OPF/2024-04-03_18-06-45/normalization_spec.pkl", "rb") as f:
    NORMALIZATION_SPEC = pickle.load(f)


# The path
SAVE_TO = f"data/gbm_simulations/controlled_power_grid/arch_1-0_1/{OPF_BBM_NAME}/{time.strftime('%Y-%m-%d_%H-%M-%S')}"

# Set the plant
SIM_PLANT = cpg_cases.case14(cc_type="data-driven", opf_bbm=OPF_BBM)
SIM_PLANT.normalization_spec = NORMALIZATION_SPEC

#%% Get initial condition and stimuli from WBM simulations
WBM_simulation_folder = "D:/projects/CPS-SenarioGeneration/data/monte_carlo/controlled_power_grid/2024-03-20_18-55-20/"
WBM_simulation_folder = Path(WBM_simulation_folder)

# Print the WBM report
with open(WBM_simulation_folder / "report.pkl", "rb") as f:
    report = pickle.load(f)

# Get the number of simulations
#%% Iterate all files and obtain realizations of the external stimuli
reload(Input)
print("Loading external stimuli realizations...")
try:
    with open("data/external_stimuli_realization_cpg.pkl", "rb") as f:
        E_realizations = pickle.load(f)

except FileNotFoundError:
    locs = [x for x in WBM_simulation_folder.iterdir() if x.is_file() and x.suffix == ".pkl" and "simulation" in x.stem]
    E_realizations = []
    for file in tqdm.tqdm(locs):
        with open(file, "rb") as f:
            data = pickle.load(f)
            df = pd.DataFrame(data["uncontrolled_inputs"], index=data["time"])
            E_realizations.append(df)

    with open("data/external_stimuli_realization_cpg.pkl", "wb") as f:
        pickle.dump(E_realizations, f)

EXTERNAL_STIMULI = Input.Input(realizations=E_realizations)
NUMBER_OF_TRIALS = len(E_realizations)

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

