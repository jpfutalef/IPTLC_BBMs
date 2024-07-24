"""
Standard simulation parameters for the CPG simulation.
"""
import dill as pickle
import numpy as np
from pathlib import Path
import time
import sys
import os

SIM_MISSION_TIME = 3600 * 24 * 4
SIM_STEP_TIME = 15 * 60
SIM_INITIAL_TIME = 0.
MAX_EXECUTION_TIME = 3600 * 8
NUMBER_OF_TRIALS = float("inf")
NOW = time.strftime('%Y-%m-%d_%H-%M-%S')

# Locations
scenario_generation_loc = Path(os.environ["SCENARIOGENERATION_ROOT"])
wbm_data_loc = Path(os.environ["MODELS_ROOT"])

# Reference simulation data
# ref_data_folder = wbm_data_loc / "data/cpg/reference_simulation"
ref_data_folder = wbm_data_loc / "data/cpg/reference_simulation"

try:
    initial_condition_loc = wbm_data_loc / "data/cpg/initial_condition.npy"
    external_stimuli_loc = wbm_data_loc / "data/cpg/external_stimuli.pkl"

    SIM_INITIAL_CONDITION = np.load(str(initial_condition_loc))
    with open(external_stimuli_loc, "rb") as f:
        EXTERNAL_STIMULI = pickle.load(f)

    # TODO load the condition disturbances into the STATE_DISTURBANCES variable

except FileNotFoundError:
    SIM_INITIAL_CONDITION = None
    EXTERNAL_STIMULI = None
    STATE_DISTURBANCES = None