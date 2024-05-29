# Imports
from greyboxmodels.cpsmodels.Plant import Plant
from simulation_parameters import *

# %% Load the plant
SIM_PLANT = Plant.load("data/gb-models/iptlc/arch_1-0_0_1.pkl")
SIM_PLANT.step_top_down = False

# %% Specify saving location
SAVE_TO = f"data/gbm-simulations/iptlc/arch_1-0_0_1/{NOW}"

# %% Run simulation
SIMULATOR.set_plant(SIM_PLANT)
SIMULATOR.set_output_folder(SAVE_TO)
SIMULATOR.simulate_references()
