import matplotlib.pyplot as plt
import dill as pickle
from pathlib import Path
import numpy as np

target_simulation = "simulation_1.pkl"

# %% Original WBM
wbm_sim_folder = Path("D:/projects/CPS-SenarioGeneration/data/cpg/MonteCarlo/2024-04-03_18-06-45/")
wbm_target_sim = wbm_sim_folder / target_simulation

with open(wbm_target_sim, "rb") as f:
    wbm_sim_data = pickle.load(f)

# %% Architecture 1
gbm_1_sim_folder = Path("data/gbm_simulations/controlled_power_grid/arch_1-0_1/2024-04-08_14-14-09")
# gbm_1_sim_folder = Path("data/gbm_simulations/controlled_power_grid/arch_1-0_1/BBM1_SimpleNet/2024-04-08_14-02-43")
gbm_1_target_sim = gbm_1_sim_folder / target_simulation

with open(gbm_1_target_sim, "rb") as f:
    gbm_1_sim_data = pickle.load(f)

# %% Architecture 2
gbm_2_sim_folder = Path("data/gbm_simulations/controlled_power_grid/arch_2-1_0/2024-04-08_02-00-59")
gbm_2_target_sim = gbm_2_sim_folder / target_simulation

with open(gbm_2_target_sim, "rb") as f:
    gbm_2_sim_data = pickle.load(f)

# %% Full BBM
full_bbm_sim_folder = Path("data/gbm_simulations/controlled_power_grid/arch_3-1_1/2024-04-08_02-00-59")
full_bbm_target_sim = full_bbm_sim_folder / target_simulation

with open(full_bbm_target_sim, "rb") as f:
    full_bbm_sim_data = pickle.load(f)

# %% Group the data in lists
sim_data_list = [wbm_sim_data, gbm_1_sim_data, gbm_2_sim_data, full_bbm_sim_data]
names_list = ["WBM", "GBM1", "GBM2", "Full BBM"]
processed_data = {x: {} for x in names_list}


# %% A function to extract inputs and outputs
def minmax_normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def get_opf_data(sim_data, output_norm=None, input_norm=None):
    # Get the inputs and outputs
    opf_inputs = np.array([x['control_center_step_data']["opf_input"] for x in sim_data['step_data']])
    opf_outputs = np.array([x['control_center_step_data']["opf_output"] for x in sim_data['step_data']])

    # Normalize the inputs and outputs
    if output_norm is not None:
        opf_outputs = np.array([minmax_normalize(x, output_norm["output_min"], output_norm["output_max"]) for x in
                                opf_outputs])

    if input_norm is not None:
        opf_inputs = np.array([minmax_normalize(x, input_norm["input_min"], input_norm["input_max"]) for x in opf_inputs])

    return opf_inputs, opf_outputs


def get_pf_data(sim_data, input_norm=None, output_norm=None):
    # Create a function that receives a path to a simulation and returns the inputs and outputs
    # Get the plant
    plant = sim_data["plant"]

    # Get the inputs and outputs
    inputs = []
    outputs = []
    for step_dict in sim_data['step_data']:
        # Input
        power_demands = step_dict['power_demands']
        opf_output = step_dict['pg_control_input']
        x_pg = step_dict['state_post_update'][plant.state_idx.power_grid]
        u_pg = plant.power_grid.get_pf_inputs(x_pg, power_demands, opf_output)
        X = np.concatenate(u_pg)

        # Output is the response
        Y = step_dict["pg_response"]

        # Filter up to column 80 to the end TODO HARDCODED!!!!
        Y = Y[:80]

        inputs.append(X)
        outputs.append(Y)

    # Concatenate the inputs and outputs
    X = np.vstack(inputs)
    Y = np.vstack(outputs)

    # Normalize the inputs and outputs
    if output_norm is not None:
        Y = np.array([minmax_normalize(y, output_norm["output_min"], output_norm["output_max"]) for y in Y])

    if input_norm is not None:
        X = np.array([minmax_normalize(x, input_norm["input_min"], input_norm["input_max"]) for x in X])

    return X, Y


# %% Get them
opf_output_norm = full_bbm_sim_data["plant"].control_center.normalization_spec
pf_output_norm = full_bbm_sim_data["plant"].power_grid.normalization_spec

for sim_data, name in zip(sim_data_list, names_list):
    processed_data[name]["OPF"] = get_opf_data(sim_data, output_norm=opf_output_norm)
    processed_data[name]["PF"] = get_pf_data(sim_data, output_norm=pf_output_norm)

# %% Plot for the OPF
num_outputs = processed_data["WBM"]["OPF"][1].shape[1]
num_cols = 5
num_rows = int(np.ceil(num_outputs / num_cols))

line_color = ["black", "orange", "green", "red"]
line_alpha = [1, 0.6, 0.6, 0.6]

fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
axs = axs.flatten()

for i in range(num_cols * num_rows):
    if i >= num_outputs:
        axs[i].axis("off")
        continue
    for k, (name, data) in enumerate(processed_data.items()):
        axs[i].plot(data["OPF"][1][:, i], label=name, color=line_color[k], alpha=line_alpha[k])

    axs[i].set_title(f"Output {i}")
    #axs[i].set_ylim(-0.1, 1.1)

# Legend at the top of the plot
fig.legend(names_list, loc="upper center", ncol=len(names_list))
fig.set_layout_engine("constrained")
fig.savefig("data/response_1.pdf")
fig.show()

# %% Pairwise comparison: WBM against the GBMs and BBM
num_outputs = processed_data["WBM"]["OPF"][1].shape[1]
num_cols = 5
num_rows = int(np.ceil(num_outputs / num_cols))

fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 5))
axs = axs.flatten()

for i in range(num_cols * num_rows):
    if i >= num_outputs:
        axs[i].axis("off")
        continue

    # WBM data
    wbm_data = processed_data["WBM"]["OPF"][1][:, i]
    for k, (name, data) in enumerate(processed_data.items()):
        if name == "WBM":
            continue
        gbm_data = data["OPF"][1][:, i]
        axs[i].plot(wbm_data, gbm_data, "o", label=name, color=line_color[k], alpha=line_alpha[k])

    #45 degree line
    axs[i].plot([-1, 2], [-1, 2], "k--")

    axs[i].set_title(f"Output {i}")
    axs[i].set_ylim(-0.1, 1.1)
    axs[i].set_xlim(-0.1, 1.1)

    # grid
    axs[i].grid()


# Legend at the top of the plot
fig.legend(names_list[1:], loc="upper center", ncol=len(names_list[1:]), bbox_to_anchor=(0.5, 1.1))
fig.set_layout_engine("constrained")
fig.savefig("data/response_2.svg", bbox_inches="tight")
fig.show()

# %% Plot for the PF
num_outputs = processed_data["WBM"]["PF"][1].shape[1]
num_cols = 8
num_rows = int(np.ceil(num_outputs / num_cols))

line_color = ["black", "orange", "green", "red"]
line_alpha = [1, 0.6, 0.6, 0.6]

fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10), sharex=True)
axs = axs.flatten()

for i in range(num_cols * num_rows):
    if i >= num_outputs:
        axs[i].axis("off")
        continue
    axs[i].set_title(f"Input {i}")
    for k, (name, data) in enumerate(processed_data.items()):
        axs[i].plot(data["PF"][1][:, i], label=name, color=line_color[k], alpha=line_alpha[k])

    axs[i].set_title(f"Output {i}")
    #axs[i].set_ylim(-0.1, 1.1)

# Legend at the bottom of the plot
fig.set_layout_engine("constrained")
fig.legend(names_list, loc="lower center", ncol=len(names_list), bbox_to_anchor=(0.5, 0))
fig.show()

# %% Pairwise comparison: WBM against the GBMs and BBM
num_outputs = processed_data["WBM"]["PF"][1].shape[1]
num_cols = 10
num_rows = int(np.ceil(num_outputs / num_cols))

fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 10))
axs = axs.flatten()

for i in range(num_cols * num_rows):
    if i >= num_outputs:
        axs[i].axis("off")
        continue

    # WBM data
    wbm_data = processed_data["WBM"]["PF"][1][:, i]
    for k, (name, data) in enumerate(processed_data.items()):
        if name == "WBM":
            continue
        gbm_data = data["PF"][1][:, i]
        axs[i].plot(wbm_data, gbm_data, "o", label=name, color=line_color[k], alpha=0.4)

    # 45 degree line
    axs[i].plot([-1, 2], [-1, 2], "k--")

    axs[i].set_title(f"Output {i}")
    axs[i].set_ylim(-0.1, 1.1)
    axs[i].set_xlim(-0.1, 1.1)

    # grid
    axs[i].grid()

# Legend at the top of the plot
fig.legend(names_list[1:], loc="upper center", ncol=len(names_list[1:]), bbox_to_anchor=(0.5, 1.1))
fig.set_layout_engine("constrained")
fig.savefig("data/response_3.png", bbox_inches="tight", dpi=600)
fig.show()
