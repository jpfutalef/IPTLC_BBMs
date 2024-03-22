import numpy as np
from pathlib import Path
import dill as pickle
import pandas as pd
import tqdm
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid, cumulative_trapezoid
import greyboxmodels.cpsmodels.Plant as Plant

def ks_statistic(ecdf_1, ecdf_2):
    """
    Compute the Kolmogorov-Smirnov statistic between two empirical cumulative distribution functions.

    Parameters
    ----------
    ecdf_1 : np.ndarray
        The first empirical cumulative distribution function.

    ecdf_2 : np.ndarray
        The second empirical cumulative distribution function.

    Returns
    -------
    float
        The Kolmogorov-Smirnov statistic.
    """
    return np.max(np.abs(ecdf_1 - ecdf_2))


def extract_state_dataframe(sim_path, exclude_from_col=None):
    """
    Open the simulation file and return the states.

    Parameters
    ----------
    sim_path : Path
        The path to the simulation file.

    exclude_from_col : int
        An int with the index of the columns to be excluded.

    Returns
    -------
    np.ndarray
        The states.
    """
    with open(sim_path, "rb") as f:
        d = pickle.load(f)

    df = pd.DataFrame(d["state"], index=d["time"])
    if exclude_from_col is not None:
        df = df.iloc[:, :exclude_from_col]

    return df


def states_from_folder(sim_folder, exclude_from_col=None):
    """
    Develops a list of dataframes with the states of the simulations.

    Parameters
    ----------
    sim_folder : Path
        The path to the folder with the simulations.

    exclude_from_col : int
        A list with the indexes of the columns to be excluded.

    Returns
    -------
    dict
        A list of dataframes with the states of the simulations.
    """
    locs = [x for x in sim_folder.iterdir() if x.is_file() and x.suffix == ".pkl" and "simulation" in x.stem]

    state_dfs = {file.stem: extract_state_dataframe(file, exclude_from_col) for file in tqdm.tqdm(locs)}

    return state_dfs


# %% Test the approach
wbm_sim_folder = "D:/projects/CPS-SenarioGeneration/data/monte_carlo/controlled_power_grid/2024-03-20_18-55-20/"
wbm_sim_folder = Path(wbm_sim_folder)

gbm_sim_folder = "D:/projects/IPTLC_BBMs/data/monte_carlo/controlled_power_grid/arch_1-0_1/BBM1_SimpleNet/2024-03-22_13-39-09"
gbm_sim_folder = Path(gbm_sim_folder)

try:
    with open("data/wbm_states.pkl", "rb") as f:
        wbm_states = pickle.load(f)

except FileNotFoundError:
    wbm_states = states_from_folder(wbm_sim_folder, exclude_from_col=80)
    with open("data/wbm_states.pkl", "wb") as f:
        pickle.dump(wbm_states, f)


try:
    with open("data/gbm_states.pkl", "rb") as f:
        gbm_states = pickle.load(f)

except FileNotFoundError:
    gbm_states = states_from_folder(gbm_sim_folder, exclude_from_col=80)
    with open("data/gbm_states.pkl", "wb") as f:
        pickle.dump(gbm_states, f)

#%% Open the plant
with open(wbm_sim_folder / "plant.pkl", "rb") as f:
    wbm_plant = pickle.load(f)

with open(gbm_sim_folder / "plant.pkl", "rb") as f:
    gbm_plant = pickle.load(f)

#%% State names
pg_state_names = Plant.get_variables_names(wbm_plant.power_grid.state_idx)
cc_state_names = Plant.get_variables_names(wbm_plant.control_center.state_idx)

#%% Remove the keys that are not in both dictionaries
# TODO THIS SHOULD NOT BE NECESSARY
keys = set(wbm_states.keys()).intersection(gbm_states.keys())
wbm_states = {key: wbm_states[key] for key in keys}
gbm_states = {key: gbm_states[key] for key in keys}

#%% Compute the KS statistic for each state
ks_dict = {}
states = list(wbm_states.values())[0].columns

for state in states:
    # Iterate times
    ks_list = []
    for t in list(wbm_states.values())[0].index:
        # Accumulate the values of this state at this time for all simulations
        wbm_values = np.array([wbm_states[key].loc[t, state] for key in keys])
        gbm_values = np.array([gbm_states[key].loc[t, state] for key in keys])

        # Compute the ECDF for each set of values
        # First, ensure both have the same bins
        n_bins = 50
        bins = np.linspace(min(np.min(wbm_values), np.min(gbm_values)),
                           max(np.max(wbm_values), np.max(gbm_values)),
                           n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bins_diff = np.diff(bins)

        # Compute the empirical pdf
        wbm_epdf, _ = np.histogram(wbm_values, bins=bins, density=True)
        gbm_epdf, _ = np.histogram(gbm_values, bins=bins, density=True)

        # Empirical CDF
        wbm_ecdf = np.cumsum(wbm_epdf) * bins_diff
        gbm_ecdf = np.cumsum(gbm_epdf) * bins_diff

        # Add zero and one to the ECDF to ensure the KS statistic is computed correctly
        wbm_ecdf = np.concatenate(([0], wbm_ecdf, [1]))
        gbm_ecdf = np.concatenate(([0], gbm_ecdf, [1]))

        # In the bins, include new min and max values
        first_diff = bins_diff[0]
        last_diff = bins_diff[-1]
        bin_centers = np.concatenate(([bin_centers[0] - first_diff], bin_centers, [bin_centers[-1] + last_diff]))

        # Compute the KS statistic
        ks = ks_statistic(wbm_ecdf, gbm_ecdf)

        # Store the KS statistic
        ks_list.append(ks)

    # Store the KS statistic for this state
    ks_dict[state] = ks_list

#%% Plot the KS statistics

# One subplot for each state
n_states = len(states)
n_cols = 8
n_rows = n_states // n_cols

fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 10), constrained_layout=True, sharex=True, sharey=True)
axs = axs.flatten()

t = list(wbm_states.values())[0].index / 3600

for i in range(n_cols * n_rows):
    ax = axs[i]
    if i >= n_states:
        ax.axis("off")
        continue
    state = states[i]
    ax.plot(t, ks_dict[state], "k")
    ax.set_title(pg_state_names[i])

    # If last row, set x label
    if i >= n_states - n_cols:
        ax.set_xlabel("Time [s]")

    # If first column, set y label
    if i % n_cols == 0:
        ax.set_ylabel("KS statistic")

    ax.set_xlim([0, t[-1]])


fig.show()

