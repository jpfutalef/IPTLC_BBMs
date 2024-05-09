from pathlib import Path
import dill as pickle
from torch_geometric import utils
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sim_folder = Path("D:/projects/CPS-SenarioGeneration/data/iptlc/MonteCarlo/2024-05-09_15-16-30")

# %% Open plant
with open(sim_folder / "plant.pkl", "rb") as f:
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

# %% Open simulations
target_simulation = Path(sim_folder, "simulation_0.pkl")
with open(target_simulation, "rb") as f:
    sim_data = pickle.load(f)

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

    # Set latencies
    for edge, lat in zip(nx_tlcn.edges, latency):
        # Make latency the only attribute
        nx.set_edge_attributes(nx_tlcn, {edge: lat}, "latency")

    # Remove the edges where piEdge is zero
    edge_remove = [i for i, edge in enumerate(nx_tlcn.edges) if piEdge[i] == 0]
    nx_tlcn.remove_edges_from(edge_remove)

    # Remove the nodes where piNode is zero
    node_remove = [i for i, node in enumerate(nx_tlcn.nodes) if piNode[i] == 0]
    nx_tlcn.remove_nodes_from(node_remove)

    return nx_tlcn


# %% Inputs
tlcn_x_idx = plant.tlc_network.state_idx
graph_data_bottom_up = []
graph_data_top_down = []
for step_data in sim_data["step_data"]:
    # bottom up
    x_tlcn_bottom_up = step_data["tlc_network_state_bottom_up"]
    nx_tlcn_bottom_up = develop_networkx_graph(plant.tlc_network, x_tlcn_bottom_up, tlcn_x_idx)
    gdata = utils.from_networkx(nx_tlcn_bottom_up, group_edge_attrs=["latency"])
    graph_data_bottom_up.append(gdata)

    # top down
    x_tlcn_top_down = step_data["tlc_network_state_top_down"]
    nx_tlcn_top_down = develop_networkx_graph(plant.tlc_network, x_tlcn_top_down, tlcn_x_idx)
    gdata = utils.from_networkx(nx_tlcn_top_down, group_edge_attrs=["latency"])
    graph_data_top_down.append(gdata)

# Dataloaders
loader_bottom_up = DataLoader(graph_data_bottom_up, batch_size=32, shuffle=False)
loader_top_down = DataLoader(graph_data_top_down, batch_size=32, shuffle=False)

#%%
gdata = utils.from_networkx(graph_list_bottom_up[0], group_edge_attrs=["latency"])

# %% Outputs
