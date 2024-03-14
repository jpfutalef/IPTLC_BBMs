import os
import warnings
import tqdm as tqdm
import numpy as np
import dill as pickle
import time
import copy
import torch

import greyboxmodels.cpsmodels.cyberphysical.ControlledPowerGrid.cases as cpg_cases
import greyboxmodels.cpsmodels.physical.electrical.cases as pg_cases
import greyboxmodels.cpsmodels.physical.electrical.data_utils as pg_data_utils
import greyboxmodels.cpsmodels.Input as Input
import greyboxmodels.cpsmodels.cyberphysical.ControlledPowerGrid.ControlledPowerGrid as cpg
import greyboxmodels.bbmcpsmodels.nn.feedforward_opf as nn_ff

warnings.filterwarnings('ignore')

# Set the working directory to the root of the project
os.chdir('/')
print(f'Working directory: {os.getcwd()}')

# In[5]:

# Simulation parameters
mission_time = 3600 * 24 * 4
dt = 15 * 60
max_exec_time = 3600 * 0.5

# Load the OPF BBM
opf_bbm_path = "models/BBM1_SimpleNet_MinMaxNormalizedOPF_20240228-144804.pt"
opf_bbm = nn_ff.BBM1_SimpleNet(51, 9)
opf_bbm.load_state_dict(torch.load(opf_bbm_path))

# Open the normalization specs
norm_specs_path = "data/OPF/20240227_195448/norm_min_max_values.pkl"
with open(norm_specs_path, "rb") as f:
    norm_specs = pickle.load(f)

# Rename the keys
norm_specs["input_min"] = norm_specs.pop("min_opf_input")
norm_specs["input_max"] = norm_specs.pop("max_opf_input")
norm_specs["output_min"] = norm_specs.pop("min_opf_output")
norm_specs["output_max"] = norm_specs.pop("max_opf_output")
norm_specs["type"] = "minmax"

# Load the case
plant = cpg_cases.case14(cc_type="data-driven",
                         opf_bbm=opf_bbm)
plant.control_center.normalization_spec = norm_specs

# Load the external stimuli
ext_sti, ext_sti_std = pg_data_utils.pq_load_from_day_curves('/data/Terna/one_week_Thursday.csv',
                                                             mission_time,
                                                             dt,
                                                             pg_cases.get_case14_loads(),
                                                             True)
ext_sti = Input.Input(ext_sti, std_dev=ext_sti_std / 2)

# Create a dictionary of the simulation parameters
sim_params = {
    "mission_time": mission_time,
    "dt": dt,
    "max_exec_time": max_exec_time
}
print(f"Simulation parameters: {sim_params}")


# In[6]:


def initial_condition(external_stimuli, cc_pg, at=0):
    cc = cc_pg.control_center
    pg = cc_pg.power_grid

    # Get external stimuli at zero
    e0 = external_stimuli.get_input(0)

    # Split into Pd and Qd
    n_load = len(pg.power_grid.load)
    Pd0 = e0[:n_load]
    Qd0 = e0[n_load:]

    # Statuses at zero are all 1
    piGen0 = pg.piGen_from_ppnet()
    piLine0 = pg.piLine_from_ppnet()
    piTrafo0 = pg.piTrafo_from_ppnet()

    piGen0[:] = 1
    piLine0[:] = 1
    piTrafo0[:] = 1

    # Voltage magnitudes and angles at zero
    vg_m0 = pg.power_grid.ext_grid.vm_pu.values
    vg_a0 = pg.power_grid.ext_grid.va_degree.values

    # Concatenate and pass to the cc
    u0_pg = cc.optimal_power_flow(Pd0, Qd0, vg_m0, vg_a0, piGen0, piLine0, piTrafo0)

    # Get the initial state by solving an optimal power flow
    x0_pg = pg.state_from_ppnet()
    u0_pf = pg.get_pf_inputs(x0_pg, e0, u0_pg)
    x0_pg = pg.power_flow(*u0_pf)  # Initial state

    return np.concatenate((x0_pg, u0_pg))


# In[7]:


# Save path
now = time.strftime("%Y%m%d_%H%M%S")
sim_folder = f"data/simulations/controlled_pg/{now}/"
input(f"Saving to {sim_folder}... press enter to continue.")

# In[8]:


# Create the folder if it does not exist
os.makedirs(sim_folder, exist_ok=True)

# Save the plant
with open(f"{sim_folder}plant.pkl", "wb") as f:
    pickle.dump(plant, f)

# Save the simulation parameters
with open(f'{sim_folder}sim_params.pkl', 'wb') as f:
    pickle.dump(sim_params, f)

# Loop
init_time = time.time()
sim_num = 0
while time.time() - init_time < max_exec_time:
    # This simulation path
    save_path = f"{sim_folder}simulation_{sim_num}.pkl"

    # Print message
    print(f"Starting simulation {sim_num} and saving to {save_path}")

    # Containers for the simulation data
    simulation_data = {
        "time": [],
        "external_stimuli": [],
        "state": [],
        "step_data": [],
        "plant": copy.deepcopy(plant)
    }

    # Simulation loop
    pbar = tqdm.tqdm(total=mission_time,
                     bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} (seconds) [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    x_prev = initial_condition(ext_sti, plant, 0)
    plant.sim_time = 0
    while plant.sim_time < mission_time:
        # Get the power demands
        e_pg = ext_sti.get_input(plant.sim_time)

        # Pass to the plant
        x = plant.step(dt, x_prev, e_pg, None)

        # Save the simulation data
        simulation_data["time"].append(plant.sim_time)
        simulation_data["external_stimuli"].append(e_pg)
        simulation_data["state"].append(x)
        simulation_data["step_data"].append(plant.step_data)

        # Update recursive variables
        x_prev = x

        # Update the progress bar
        pbar.update(dt)

    pbar.close()

    # Save the simulation data
    with open(save_path, "wb") as f:
        pickle.dump(simulation_data, f)

    # Increment the simulation number
    sim_num += 1

# Print message
print(f"Finished {sim_num} simulations")
