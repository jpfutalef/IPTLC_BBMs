import os
import warnings
import tqdm as tqdm
import numpy as np
import dill as pickle
import time
import copy
import torch

import greyboxmodels.models.cyberphysical.CPG.cases as cpg_cases
import greyboxmodels.models.physical.electrical.cases as pg_cases
import greyboxmodels.models.physical.electrical.data_utils as pg_data_utils
import greyboxmodels.modelbuild.Input as Input
import greyboxmodels.models.cyberphysical.CPG.ControlledPowerGrid as cpg
import greyboxmodels.bbmcpsmodels.nn.feedforward_opf as nn_ff

warnings.filterwarnings('ignore')

# Set the working directory to the root of the project
os.chdir('D:/projects/IPTLC_BBMs/')
print(f'Working directory: {os.getcwd()}')

# In[5]:

# Simulation parameters
mission_time = 3600 * 24 * 4
dt = 15 * 60
max_exec_time = 120

# Save path
now = time.strftime("%Y%m%d_%H%M%S")
sim_folder = f"sim_data/simulations/controlled_pg/{now}/"

# Load the OPF BBM
opf_bbm_path = "models\BBM1_SimpleNet_MinMaxNormalizedOPF_20240314-114822.pt"
opf_bbm = nn_ff.BBM1_SimpleNet(52, 10)
opf_bbm.load_state_dict(torch.load(opf_bbm_path))

# Open the normalization specs
norm_specs_path = "sim_data/OPF/20240311_011412/norm_min_max_values.pkl"
with open(norm_specs_path, "rb") as f:
    norm_specs = pickle.load(f)

# Rename the keys
norm_specs["input_min"] = norm_specs.pop("min_input")
norm_specs["input_max"] = norm_specs.pop("max_input")
norm_specs["output_min"] = norm_specs.pop("min_output")
norm_specs["output_max"] = norm_specs.pop("max_output")
norm_specs["type"] = "minmax"

# Load the case
plant = cpg_cases.case14(cc_type="sim_data-driven",
                         opf_bbm=opf_bbm)
plant.control_center.normalization_spec = norm_specs

# Load the external stimuli
ext_sti, ext_sti_std = pg_data_utils.pq_load_from_day_curves('D:/projects/Hierarchical_CPS_models/data/Terna/one_week_Thursday.csv',
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
    pg = cc_pg.ppnet

    # Get external stimuli at zero
    e0 = external_stimuli.get_input(at)

    # Split into Pd and Qd
    pg_e_idx = pg.uncontrolled_inputs_idx
    Pd0 = e0[pg_e_idx.Pd]
    Qd0 = e0[pg_e_idx.Qd]

    # Statuses at zero are all 1
    piGen0 = pg.piGen_from_ppnet()
    piLine0 = pg.piLine_from_ppnet()
    piTrafo0 = pg.piTrafo_from_ppnet()

    piGen0[:] = 1
    piLine0[:] = 1
    piTrafo0[:] = 1

    # Get maximum bus values
    max_v_bus = pg.ppnet.bus.max_vm_pu.values

    # Get those from the generators
    gen_buses = pg.ppnet.gen.bus.values

    # Filter the maximum values using the buses
    vg_m0 = max_v_bus[gen_buses]

    # Do an OPF run
    u0_pg = cc.optimal_power_flow(Pd0, Qd0, vg_m0, piGen0, piLine0, piTrafo0)

    # Get the initial state by solving an optimal power flow
    x0_pg = pg.state_from_ppnet()
    _, _, Pg0_gen, Vm0_gen, _, _, _ = pg.get_pf_inputs(x0_pg, e0, u0_pg)
    x0_pg = pg.power_flow(Pd0, Qd0, Pg0_gen, Vm0_gen, piGen0, piLine0, piTrafo0)  # Initial state

    return np.concatenate((x0_pg, u0_pg))


# In[7]:
input(f"Saving to {sim_folder}... press enter to continue.")

# Create the folder if it does not exist
os.makedirs(sim_folder, exist_ok=True)

# Save the plant
with open(f"{sim_folder}plant.pkl", "wb") as f:
    pickle.dump(plant, f)

# Save the simulation parameters
with open(f'{sim_folder}sim_params.pkl', 'wb') as f:
    pickle.dump(sim_params, f)

init_time = time.time()
sim_num = 0
while time.time() - init_time < max_exec_time:
    # This simulation path
    save_path = f"{sim_folder}simulation_{sim_num}.pkl"

    # Print message
    print(f"Starting simulation {sim_num} and saving to {save_path}")

    # Containers for the simulation sim_data
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

    x_prev = cpg.initial_condition(ext_sti, plant, 0)
    t_miss = 0
    loop_time = np.inf
    while t_miss < mission_time:
        tinit_loop = time.time()
        # Update the time
        t_miss += dt

        # Get the power demands
        e_pg = ext_sti.get_input(t_miss)

        # Pass to the plant
        x = plant.step(dt, x_prev, e_pg, None)

        # Save the simulation sim_data
        simulation_data["time"].append(t_miss)
        simulation_data["external_stimuli"].append(e_pg.copy())
        simulation_data["state"].append(x.copy())
        simulation_data["step_data"].append(plant.step_data.copy())

        # Update recursive variables
        x_prev = x

        # Update the progress bar
        pbar.update(dt)

        tend_loop = time.time()
        loop_time = tend_loop - tinit_loop

    pbar.close()

    # Save the simulation sim_data
    with open(save_path, "wb") as f:
        pickle.dump(simulation_data, f)

    # Increment the simulation number
    sim_num += 1

# Print message
print(f"Finished {sim_num} simulations")