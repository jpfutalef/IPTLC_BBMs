from pathlib import Path
import numpy as np

import greyboxmodels.scenariogeneration.DeterministicSimulator as simulation
from greyboxmodels.modelbuild.Plant import Plant
from greyboxmodels.models.cyberphysical.CPG import ControlledPowerGrid as CPG

# Specify locations
#plant_location = Path("data/gb-models/cpg/arch1_0-1.pkl")
plant_location = Path("data/gb-models/cpg/arch2_1-0.pkl")
#plant_location = Path("data/gb-models/cpg/arch3_1-1.pkl")
#plant_location = Path("../Hierarchical_CPS_models/data/wbm-models/cpg_ieee14-deterministic_wbm.pkl")
plant: CPG.ControlledPowerGrid = Plant.load(plant_location)

ref_data_folder = Path("data/dns_scenarios/cpg/")
output_folder = ref_data_folder / plant_location.stem

#%% Simulator
sim = simulation.Simulator(plant,
                           output_folder=output_folder,
                           )

#%% Specify the indices in the state vector that are disturbances
pg_idx = plant.state_idx.power_grid
x_idx = plant.power_grid.state_idx
piGen = pg_idx[x_idx.piGen]
piLine = pg_idx[x_idx.piLine]
piTrafo = pg_idx[x_idx.piTrafo]

pi_idx = np.concatenate([piGen, piLine, piTrafo])

#%% Simulate
for file in ref_data_folder.iterdir():
    try:
        sim_data = sim.simulate_reference(file,
                                          state_idx=pi_idx,
                                          )

    except Exception as e:
        print(f"Error: {e}")
        continue
