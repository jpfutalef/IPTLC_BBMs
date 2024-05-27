"""
7 tlc nodes

Author: Juan-Pablo Futalef
"""


def case(e2e_metamodels):
    """
    Create a TLCN
    :return:
    """
    import numpy as np
    import greyboxmodels.cpsmodels.cyber.TLCN.TLCNetwork as TLCN

    tlc_adjacency = np.array([[0, 1, 0, 0, 1, 1, 0],
                              [1, 0, 1, 0, 0, 0, 1],
                              [0, 1, 0, 0, 0, 0, 1],
                              [0, 0, 0, 0, 1, 0, 1],
                              [1, 0, 0, 1, 0, 1, 0],
                              [1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 1, 1, 0, 0, 0]])

    xy_coords = [(97, 108),
                 (93, 232),
                 (337, 288),
                 (350, 146),
                 (208, 142),
                 (251, 47),
                 (208, 280)]

    # These are the same for all nodes/edges
    min_latency = 10e-3
    max_latency = 30e-3
    packet_latency_tolerance = 80e-3
    latency_increment = 1e-3
    p_drop_node = 1e-8
    p_drop_edge = 0.02

    # Create the TLCN nodes
    tlc_nodes = [TLCN.TLCNode(i, xy_coords[i][0], xy_coords[i][1]) for i in range(tlc_adjacency.shape[0])]

    # Instantiate the TLCN
    TLC_NETWORK = TLCN.DataDrivenTLCN(tlc_nodes,
                                      tlc_adjacency,
                                      packet_latency_tolerance,
                                      min_latency,
                                      max_latency,
                                      latency_increment,
                                      p_drop_node,
                                      p_drop_edge,
                                      end_to_end_metamodels=e2e_metamodels)

    return TLC_NETWORK


# %% Run the main loop
if __name__ == '__main__':
    import xgboost as xgb

    # Specify locations of the files
    save_to = "data/bb-models/tlcn_case7_wbm.pkl"

    # Location of the metamodel
    mm_loc = "models/TLCN/RESTART_model.json"

    # Load
    mm = xgb.XGBClassifier()
    mm.load_model(mm_loc)

    # Add to the dictionary of origin-destination pairs
    d = {(5, 6): mm, (6, 5): mm}

    # Create the case
    plant = case(d)

    # Save the model
    plant.save(save_to)
