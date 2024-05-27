"""
Standard simulation parameters for the CPG simulation.
"""
import dill as pickle
import numpy as np
from pathlib import Path
import time

SIM_MISSION_TIME = 3600 * 24 * 4
SIM_STEP_TIME = 15 * 60
SIM_INITIAL_TIME = 0.
MAX_EXECUTION_TIME = 3600 * 8
NUMBER_OF_TRIALS = float("inf")
NOW = time.strftime('%Y-%m-%d_%H-%M-%S')

# WBM data location
wbm_data_loc = Path("D:/projects/CPS-SenarioGeneration")

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