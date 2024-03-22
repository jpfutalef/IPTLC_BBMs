def exec_time(sim_dict: dict):
    # Get the total execution time
    return sim_dict["total_execution_time"]


def exec_time_array(sim_dict: dict):
    # Get the execution time array
    return sim_dict["execution_time_array"]


def sim_time_array(sim_dict: dict):
    # Get the simulation time array
    return sim_dict["time"]


# %% Test the computational load
# if __name__ == '__main__':
from pathlib import Path
import dill as pickle
import tqdm

#%% The WBM
sim_folder = "D:/projects/CPS-SenarioGeneration/data/monte_carlo/controlled_power_grid/2024-03-20_18-55-20/"
sim_folder = Path(sim_folder)
locs = [x for x in sim_folder.iterdir() if x.is_file() and x.suffix == ".pkl" and "simulation" in x.stem]

l_list = []
l_array_list = []
t_array_list = []
for file in tqdm.tqdm(locs):
    with open(file, "rb") as f:
        d = pickle.load(f)

    # Compute the computational load
    l = exec_time(d)
    l_array = exec_time_array(d)
    sim_time = sim_time_array(d)
    print(f"Total time for {file.stem}: {l}")
    l_list.append(l)
    l_array_list.append(l_array)
    t_array_list.append(sim_time)

print(f"Averaged computational load (WBM): {sum(l_list) / len(l_list)}")

#%% The BBM
sim_folder = "D:/projects/IPTLC_BBMs/data/monte_carlo/controlled_power_grid/arch_1-0_1/BBM1_SimpleNet/2024-03-22_10-41-19"
sim_folder = Path(sim_folder)
locs = [x for x in sim_folder.iterdir() if x.is_file() and x.suffix == ".pkl" and "simulation" in x.stem]

bbm_l_list = []
bbm_l_array_list = []
bbm_t_array_list = []
for file in tqdm.tqdm(locs):
    with open(file, "rb") as f:
        d = pickle.load(f)

    # Compute the computational load
    l = exec_time(d)
    l_array = exec_time_array(d)
    sim_time = sim_time_array(d)
    print(f"Total time for {file.stem}: {l}")
    bbm_l_list.append(l)
    bbm_l_array_list.append(l_array)
    bbm_t_array_list.append(sim_time)

print(f"Averaged computational load (BBM): {sum(bbm_l_list) / len(bbm_l_list)}")


# %% Calculations
import matplotlib.pyplot as plt
import numpy as np

# Plot the execution time arrays
fig, ax = plt.subplots(figsize=(5, 5))
for t, l, bbm_l in zip(t_array_list, l_array_list, bbm_l_array_list):
    l = [x-l[0] for x in l]
    bbm_l = [x-bbm_l[0] for x in bbm_l]
    t = [x/3600. for x in t]
    ax.plot(t, l, color="gray", alpha=0.3)
    ax.plot(t, bbm_l, color="blue", alpha=0.3)
    ax.set_xlabel("Simulation time [h]")
    ax.set_ylabel("Execution time [s]")
    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim(0, 500)
# Horizontal grids
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Figure title
fig.suptitle("Execution time for WBM and GBM simulation")
fig.show()

#%% Total averages and std
wbm_total_avg = sum([l[-1] - l[0] for l in l_array_list]) / len(l_array_list)
wbm_total_std = np.std([l[-1] - l[0] for l in l_array_list])
print(f"Total average execution time for WBM: {wbm_total_avg}")
print(f"Total std execution time for WBM: {wbm_total_std}")

bbm_total_avg = sum([l[-1] - l[0] for l in bbm_l_array_list]) / len(bbm_l_array_list)
bbm_total_std = np.std([l[-1] - l[0] for l in bbm_l_array_list])
print(f"Total average execution time for BBM: {bbm_total_avg}")
print(f"Total std execution time for BBM: {bbm_total_std}")
