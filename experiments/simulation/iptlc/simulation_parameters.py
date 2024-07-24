"""
Standard simulation parameters for the simulation.
"""
import dill as pickle
import numpy as np
from pathlib import Path
import time

from greyboxmodels.scenariogeneration import DeterministicSimulator

SIM_MISSION_TIME = 3600 * 24 * 4
SIM_STEP_TIME = 15 * 60
SIM_INITIAL_TIME = 0.
MAX_EXECUTION_TIME = 3600 * 10

# WBM data location
wbm_repo_root = Path("D:/projects/CPS-SenarioGeneration")
wbm_simulations = wbm_repo_root / "data/iptlc/MonteCarlo/2024-05-09_15-16-30/"
NOW = wbm_simulations.name

try:
    initial_condition_loc = wbm_repo_root / "data/iptlc/initial_condition.npy"
    SIM_INITIAL_CONDITION = np.load(str(initial_condition_loc))

    external_stimuli_loc = wbm_repo_root / "data/iptlc/external_stimuli.pkl"
    with open(external_stimuli_loc, "rb") as f:
        EXTERNAL_STIMULI = pickle.load(f)

except FileNotFoundError:
    SIM_INITIAL_CONDITION = None
    EXTERNAL_STIMULI = None

# %% Set up the plant
wbm_plant_path = wbm_simulations / "plant.pkl"
with open(wbm_plant_path, "rb") as f:
    wbm_plant = pickle.load(f)

# %% Indices to force
controlled_inputs_idx = None
external_stimuli_idx = "all"

# The the target state indices for the power grid
pg = wbm_plant.power_grid
pg_idx = wbm_plant.state_idx.power_grid
pi_pg_idx = np.concatenate((pg.state_idx.piGen,
                            pg.state_idx.piLine,
                            pg.state_idx.piTrafo
                            ))
target_pg = pg_idx[pi_pg_idx]

# The the target state indices for the telecommunication network
tlcn = wbm_plant.tlc_network
tlcn_top_down_idx = wbm_plant.state_idx.tlc_network_top_down
tlcn_bottom_up_idx = wbm_plant.state_idx.tlc_network_bottom_up
td_idx = tlcn.state_idx.Td
pi_tlcn_idx = np.concatenate((tlcn.state_idx.piNode, tlcn.state_idx.piEdge))
tlcn_idx = np.concatenate((td_idx, pi_tlcn_idx))
target_tlcn = np.concatenate((tlcn_bottom_up_idx[tlcn_idx], tlcn_top_down_idx[tlcn_idx]))

# Concatenate the target states
state_idx = np.concatenate((target_pg, target_tlcn))

# %% Create the simulator
SIMULATOR = DeterministicSimulator.SimulatorWithReference(reference_folder=wbm_simulations,
                                                          state_idx=state_idx,
                                                          external_stimuli_idx=external_stimuli_idx,
                                                          control_inputs=controlled_inputs_idx)
SIMULATOR.x0 = SIM_INITIAL_CONDITION
SIMULATOR.mission_time = SIM_MISSION_TIME
SIMULATOR.step_time = SIM_STEP_TIME
SIMULATOR.max_exec_time = MAX_EXECUTION_TIME
SIMULATOR.external_stimuli = EXTERNAL_STIMULI
