from pathlib import Path
import dill as pickle
import tqdm


def exec_time(sim_dict: dict):
    # Get the total execution time
    return sim_dict["total_execution_time"]


def exec_time_array(sim_dict: dict):
    # Get the execution time array
    exec_time = sim_dict["execution_time_array"]

    # Minus the first element
    exec_time_array = [x - exec_time[0] for x in exec_time]
    return exec_time_array


def sim_time_array(sim_dict: dict):
    # Get the simulation time array
    return sim_dict["time"]


def process_folder(folder):
    folder = Path(folder)
    # Get the execution time for all simulations in a folder
    locs = [x for x in folder.iterdir() if x.is_file() and x.suffix == ".pkl" and "simulation" in x.stem]

    info = {"folder": folder,
            "execution_times": [],
            "simulation_times": [],
            "total_execution_time": []
            }
    for file in tqdm.tqdm(locs):
        with open(file, "rb") as f:
            d = pickle.load(f)

        # Compute the computational load
        info["execution_times"].append(exec_time_array(d))
        info["simulation_times"].append(sim_time_array(d))
        info["total_execution_time"].append(exec_time(d))

    return info


# %% Test the computational load
wbm_info = process_folder("D:/projects/CPS-SenarioGeneration/sim_data/cpg/MonteCarlo/2024-04-03_18-06-45/")
gbm1_info = process_folder("sim_data/gbm_simulations/controlled_power_grid/arch_1-0_1/2024-04-08_14-14-09")
gbm2_info = process_folder("sim_data/gbm_simulations/controlled_power_grid/arch_2-1_0/BBM1_SimpleNet/2024-04-08_02-00-59")
gbm3_info = process_folder("sim_data/gbm_simulations/controlled_power_grid/arch_3-1_1/2024-04-08_02-00-59")

#%% Group
info_list = [wbm_info, gbm1_info, gbm2_info, gbm3_info]
names = ["WBM", "GBM1", "GBM2", "Full BBM"]

# Save
with open("sim_data/voi_losses/time_sim_time_array.pkl", "wb") as f:
    pickle.dump(info_list, f)

# %% Plots
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

cmap = matplotlib.colormaps['tab10']

# Plot the execution time arrays
fig, ax = plt.subplots(figsize=(5, 7), constrained_layout=True, dpi=600)

for color_idx, info in enumerate(info_list):
    for t, l in zip(info["simulation_times"], info["execution_times"]):
        t = np.array(t) / 3600
        ax.plot(t, l, color=cmap(color_idx/10), alpha=0.3)
        ax.set_xlabel("Simulation time [h]")
        ax.set_ylabel("Execution time [s]")
        ax.set_xlim([t[0], t[-1]])
        ax.set_ylim(0, 500)

# Horizontal grids
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Figure title
fig.suptitle("Execution time for WBM and GBM simulation")
fig.show()

#%% Statistics
# Store in a dataframe
import pandas as pd

d = {}

for info, name in zip(info_list, names):
    print(f"Folder: {info['folder']}")
    mean = np.mean(info["total_execution_time"])
    std = np.std(info["total_execution_time"])
    print(f"Total execution time: {mean:.2f} ± {std:.2f} s")

    # Store in dict
    d[name] = {"L1": mean}
    print("")

df = pd.DataFrame(d).T

#%% Save
df.to_csv("sim_data/voi_losses/l1_computational_load/l1_computational_load.csv")



