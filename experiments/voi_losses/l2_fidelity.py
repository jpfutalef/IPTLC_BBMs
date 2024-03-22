import numpy as np
from pathlib import Path
import dill as pickle
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

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

gbm_sim_folder = "D:/projects/IPTLC_BBMs/data/monte_carlo/controlled_power_grid/arch_1-0_1/BBM1_SimpleNet/2024-03-22_10-41-19"
gbm_sim_folder = Path(gbm_sim_folder)

# Get the list of states
gbm_states = states_from_folder(gbm_sim_folder, exclude_from_col=80)
wbm_states = states_from_folder(wbm_sim_folder, exclude_from_col=80)

#%% Remove the keys that are not in both dictionaries
keys = set(wbm_states.keys()).intersection(gbm_states.keys())
wbm_states = {key: wbm_states[key] for key in keys}
gbm_states = {key: gbm_states[key] for key in keys}

# Save to avoid wasting time
with open("data/wbm_states.pkl", "wb") as f:
    pickle.dump(wbm_states, f)

with open("data/gbm_states.pkl", "wb") as f:
    pickle.dump(gbm_states, f)

#%% Open
with open("data/wbm_states.pkl", "rb") as f:
    wbm_states = pickle.load(f)

with open("data/gbm_states.pkl", "rb") as f:
    gbm_states = pickle.load(f)

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
        bins = np.linspace(min(np.min(wbm_values), np.min(gbm_values)),
                           max(np.max(wbm_values), np.max(gbm_values)),
                           100)

        # Compute the ECDFs
        wbm_ecdf = np.histogram(wbm_values, bins=bins, density=True)[0].cumsum()
        gbm_ecdf = np.histogram(gbm_values, bins=bins, density=True)[0].cumsum()

        # Compute the KS statistic
        ks = ks_statistic(wbm_ecdf, gbm_ecdf)

        # Store the KS statistic
        ks_list.append(ks)

    # Store the KS statistic for this state
    ks_dict[state] = ks_list



