# Imports
from greyboxmodels.modelbuild.Plant import Plant
from simulation_parameters import *

#%% Load the plant
SIM_PLANT = Plant.load("data/gb-models/iptlc/arch_7-1_1_1.pkl")
SIM_PLANT.step_top_down = False

#%% Specify saving location
SAVE_TO = f"data/gbm-simulations/iptlc/arch_7-1_1_1/{NOW}"

# %% Run simulation
SIMULATOR.set_plant(SIM_PLANT)
SIMULATOR.set_output_folder(SAVE_TO)
SIMULATOR.simulate_references()
