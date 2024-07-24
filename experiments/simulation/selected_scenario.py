from pathlib import Path

import numpy as np
from greyboxmodels.cpsmodels.Plant import Plant
import greyboxmodels.scenariogeneration.DeterministicSimulator as simulation

from greyboxmodels.cpsmodels.cyberphysical.ControlledPowerGrid import ControlledPowerGrid as CPG

# Catch all warnins as errors
import warnings
warnings.simplefilter("error")

# Specify locations
plant: CPG.ControlledPowerGrid = Plant.load("data/gb-models/cpg/arch1_0-1.pkl")

ref_data_folder = Path("data/dns_scenarios/cpg/")
output_folder = Path("data/dns_scenarios/cpg/arch1/")

#%% Simulator
sim = simulation.Simulator(plant,
                           output_folder=output_folder,
                           )

#%% Specify the indices in the state vector that are disturbances
pg_idx = plant.power_grid.state_idx
piGen = plant.state_idx[pg_idx.piGen]
piLine = plant.state_idx[pg_idx.piLine]
piTrafo = plant.state_idx[pg_idx.piTrafo]

pi_idx = np.concatenate([piGen, piLine, piTrafo])

plant.state_disturbances_idx = pi_idx

#%% Simulate
for file in ref_data_folder.iterdir():
    try:
        sim.simulate_reference(file)

    except Exception as e:
        print(f"Error: {e}")
        continue
