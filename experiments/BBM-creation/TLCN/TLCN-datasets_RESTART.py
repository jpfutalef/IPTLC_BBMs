import os
from pathlib import Path
import dill as pickle
import numpy as np
from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import time
import torch
import tqdm
import sys

#%% Specify locations
# sim_folder = Path("D:/projects/CPS-SenarioGeneration/data/iptlc/MonteCarlo/2024-05-09_15-16-30")
sim_folder = Path("/data/iptlc/RESTART/if1_dynamic_network_level_vulnerability/2024-05-09_15-09-08/scenarios")

# Create the output folder
dataset_id = time.strftime("%Y-%m-%d_%H-%M-%S")
output_folder = Path(f"data/IO-datasets/TLCN/RESTART_data/{dataset_id}")
os.makedirs(output_folder, exist_ok=True)

# Open plant
plant_path = sim_folder.parent / "plant.pkl"

#%% Info.txt file
info_file = Path(output_folder, "info.txt")
str_info = f"""Dataset for the TLCN created from the RESTART simulations
    - Plant: {plant_path}
    - Simulation folder: {sim_folder}
    - Output folder: {output_folder}
"""

with open(info_file, "w") as f:
    f.write(str_info)

# %% Open plant
with open(plant_path, "rb") as f:
    plant = pickle.load(f)

# %% TLCN drawing
import networkx as nx
import matplotlib.pyplot as plt

net = plant.tlc_network.base_graph
pos = plant.tlc_network.get_node_coordinates()

# lower 2
pos_2 = list(pos[2])
pos_2[1] -= 50
pos[2] = pos_2

# Draw the network
fig, ax = plt.subplots(figsize=(3, 3), dpi=500)

nx.draw_networkx_nodes(net, pos, ax=ax, node_color="w", edgecolors="k")
nx.draw_networkx_nodes(net, pos, ax=ax, node_color="lightgrey", edgecolors="k", nodelist=[6])
nx.draw_networkx_edges(net, pos, ax=ax, arrows=True, arrowstyle="<->")
nx.draw_networkx_labels(net, pos, ax=ax)

# remove the frame
# ax.axis("off")

# title
fig.suptitle("Base TLCN")

plt.tight_layout()
plt.show()

# %% Create the dataset
"""
What's happening?
From the sim_data, we obtain the state variables of the TLCN: we will use them to create the input graph data for
the GNN.

We consider a multi-output approach: one output is the result of one transmission, i.e., 11 outputs.

Since the outputs are independent, we can train 11 models, one for each output, using the same input data.
"""


def develop_networkx_graph(tlc_network, x_tlcn, tlcn_idx):
    net = tlc_network.base_graph
    nx_tlcn = net.copy()

    # Get the features from the state
    piNode = x_tlcn[tlcn_idx.piNode]
    piEdge = x_tlcn[tlcn_idx.piEdge]
    latency = x_tlcn[tlcn_idx.Td]

    # Clear edge attributes
    for (u, v, d) in nx_tlcn.edges(data=True):
        d.clear()

    # Clear node attributes
    for n, d in nx_tlcn.nodes(data=True):
        d.clear()

    # Set edge features
    for edge, lat, status in zip(nx_tlcn.edges, latency, piEdge):
        nx.set_edge_attributes(nx_tlcn, {edge: {"latency": lat / .03, "status": status}})

    # Set node features
    for node, status in zip(nx_tlcn.nodes, piNode):
        nx.set_node_attributes(nx_tlcn, {node: {"status": status}})

    return nx_tlcn


def develop_array_dataset(x_tlcn, tlcn_idx, norm_coeff_latency=0.03):
    # Get the features from the state
    piNode = x_tlcn[tlcn_idx.piNode]
    piEdge = x_tlcn[tlcn_idx.piEdge]
    latency = x_tlcn[tlcn_idx.Td]

    # Concatenate the features into a single array
    x = np.concatenate((latency / norm_coeff_latency, piNode, piEdge))

    return x


# %% Open simulations
# target_simulation = Path(sim_folder, "simulation_0.pkl")
sim_files = [x for x in sim_folder.iterdir() if x.suffix == ".pkl" and "simulation" in x.stem]

# %% Dataset for GNN
# data_list_bottom_up = []
# data_list_top_down = []
#
# for target_simulation in tqdm.tqdm(sim_files):
#     with open(target_simulation, "rb") as f:
#         sim_data = pickle.load(f)
#
#     # Inputs and outputs
#     tlcn_x_idx = plant.tlc_network.state_idx
#     for step_data in sim_data["step_data"]:
#         # bottom up
#         x_tlcn_bottom_up = step_data["tlc_network_state_bottom_up"]
#         nx_tlcn_bottom_up = develop_networkx_graph(plant.tlc_network, x_tlcn_bottom_up, tlcn_x_idx)
#
#         data_bottom_up = utils.from_networkx(nx_tlcn_bottom_up,
#                                              group_node_attrs=["status"],
#                                              group_edge_attrs=["latency", "status"])
#         y_bottom_up = [int(x["success"]) for x in
#                        step_data["bottom_up_transmission_result"].values()]  # This is a boolean
#         data_bottom_up.y = torch.tensor(y_bottom_up)
#         data_list_bottom_up.append(data_bottom_up)
#
#         # top down
#         x_tlcn_top_down = step_data["tlc_network_state_top_down"]
#         nx_tlcn_top_down = develop_networkx_graph(plant.tlc_network, x_tlcn_top_down, tlcn_x_idx)
#
#         data_top_down = utils.from_networkx(nx_tlcn_top_down,
#                                             group_node_attrs=["status"],
#                                             group_edge_attrs=["latency", "status"])
#         y_top_down = torch.tensor([int(x["success"]) for x in step_data["top_down_transmission_result"].values()])
#         data_top_down.y = y_top_down
#         data_list_top_down.append(data_top_down)
#
# # Save
# now = time.strftime("%Y-%m-%d_%H-%M-%S")
# output_folder = Path(f"data/IO-datasets/TLCN/{now}")
#
# os.makedirs(output_folder, exist_ok=True)
#
# torch.save(data_list_bottom_up, output_folder / "dataset_bottom_up.pt")
# torch.save(data_list_top_down, output_folder / "dataset_top_down.pt")

# %% Dataset for standard classifier
data_list_bottom_up = []
data_list_top_down = []

for target_simulation in tqdm.tqdm(sim_files, file=sys.stdout):
    with open(target_simulation, "rb") as f:
        sim_data = pickle.load(f)

    # Inputs and outputs
    tlcn_x_idx = plant.tlc_network.state_idx
    for step_data in sim_data["step_data"]:
        # bottom up
        x_bottom_up = develop_array_dataset(step_data["tlc_network_state_bottom_up"],
                                            tlcn_x_idx)
        y_bottom_up = [int(x["success"]) for x in step_data["bottom_up_transmission_result"].values()]  # This is a boolean

        x_bottom_up = torch.tensor(x_bottom_up, dtype=torch.float32)
        y_bottom_up = torch.tensor(y_bottom_up, dtype=torch.float32)

        data_list_bottom_up.append((x_bottom_up, y_bottom_up))

        # top down
        x_top_down = develop_array_dataset(step_data["tlc_network_state_top_down"],
                                           tlcn_x_idx)
        y_top_down = [int(x["success"]) for x in step_data["top_down_transmission_result"].values()]

        x_top_down = torch.tensor(x_top_down, dtype=torch.float32)
        y_top_down = torch.tensor(y_top_down, dtype=torch.float32)

        data_list_top_down.append((x_top_down, y_top_down))

#%% Save
print(f"Saving to:\n    {output_folder}")
torch.save(data_list_bottom_up, output_folder / "dataset_array_bottom_up.pt")
torch.save(data_list_top_down, output_folder / "dataset_array_top_down.pt")
