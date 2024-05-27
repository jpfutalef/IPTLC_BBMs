"""
Standard simulation parameters for the simulation.
"""
import dill as pickle
import numpy as np
from pathlib import Path
import time

SIM_MISSION_TIME = 3600 * 24 * 4
SIM_STEP_TIME = 15 * 60
SIM_INITIAL_TIME = 0.
MAX_EXECUTION_TIME = 3600 * 10
NOW = time.strftime('%Y-%m-%d_%H-%M-%S')

# WBM data location
wbm_repo_root = Path("D:/projects/CPS-SenarioGeneration")
wbm_simulations = wbm_repo_root / "data/iptlc/MonteCarlo/2024-05-09_15-16-30/"

try:
    initial_condition_loc = wbm_repo_root / "data/iptlc/initial_condition.npy"
    external_stimuli_loc = wbm_repo_root / "data/iptlc/external_stimuli.pkl"

    SIM_INITIAL_CONDITION = np.load(str(initial_condition_loc))

    with open(external_stimuli_loc, "rb") as f:
        EXTERNAL_STIMULI = pickle.load(f)

except FileNotFoundError:
    SIM_INITIAL_CONDITION = None
    EXTERNAL_STIMULI = None
    STATE_DISTURBANCES = None
