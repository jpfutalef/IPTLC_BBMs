from greyboxmodels.modelbuild.Plant import Plant
from greyboxmodels.scenariogeneration.MonteCarlo import MonteCarlo
from simulation_parameters import *

#%% Load the plant
SIM_PLANT = Plant.load("data/gb-models/cpg/arch2_1-0.pkl")

#%% Specify saving location
SAVE_TO = f"data/gbm-simulations/cpg/arch2_1-0/{NOW}"

# %% Run simulation
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
