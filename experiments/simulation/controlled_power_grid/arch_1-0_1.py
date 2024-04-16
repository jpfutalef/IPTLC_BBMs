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
import greyboxmodels.cpsmodels.physical.electrical.cases as pg_cases
from greyboxmodels.scenariogeneration.MonteCarlo import MonteCarlo
import greyboxmodels.bbmcpsmodels.physical.feedforward_nn_pf as pf_bbm
from greyboxmodels.bbmcpsmodels.cyber import feedforward_nn_opf as opf_bbm
import greyboxmodels.cpsmodels.cyberphysical.ControlledPowerGrid.ControlledPowerGrid as CPG
import greyboxmodels.cpsmodels.cyber.ControlCenter as CC

warnings.filterwarnings('ignore')

# Simulation parameters
SIM_MISSION_TIME = 3600 * 24 * 4
SIM_STEP_TIME = 15 * 60
SIM_INITIAL_TIME = 0.
MAX_EXECUTION_TIME = 3600 * 8

WBM_simulation_folder = "D:/projects/CPS-SenarioGeneration/data/cpg/MonteCarlo/2024-04-03_18-06-45"

#%% BBM Power grid
POWER_GRID = pg_cases.case14()


#%% BBM Control center
# Load the case
OPF_BBM = opf_bbm.BBM1_SimpleNet(52, 10)
#OPF_BBM.load_state_dict(torch.load("models/BBM1_SimpleNet_MinMaxNormalizedOPF_20240321-163701.pt"))
OPF_BBM.load_state_dict(torch.load("models/OPF/BBM1_SimpleNet_OPF_2024-04-03_18-06-45_20240408-135438.pt"))

# Get the normalization spec
with open("data/IO-datasets/OPF/2024-04-03_18-06-45/normalization_spec.pkl", "rb") as f:
    NORMALIZATION_SPEC_OPF = pickle.load(f)

CONTROL_CENTER = CC.DataDrivenControlCenter(POWER_GRID, OPF_BBM, NORMALIZATION_SPEC_OPF)

#%% Create the Controlled Power Grid
SIM_PLANT = CPG.ControlledPowerGrid(POWER_GRID, CONTROL_CENTER)

# The path
SAVE_TO = f"data/gbm_simulations/controlled_power_grid/arch_1-0_1/{time.strftime('%Y-%m-%d_%H-%M-%S')}"


#%% Get initial condition and stimuli from WBM simulations
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

