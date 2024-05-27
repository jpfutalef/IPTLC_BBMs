"""
THis script creates a data-driven model for the IEEE 14 power grid and saves them to a file.

Author: Juan-Pablo Futalef

"""


def setup_ppnet():
    """
    Create a case 14 power grid, returning the given type
    @return: the case
    """
    import numpy as np
    import pandapower as pp
    import pandapower.networks as nets

    # Instance from the PandaPower case 14 using the given type
    ppnet = nets.case14()

    # ----- GENERATORS -----
    # Move  the gen in ext_grid to the gen table
    pp.replace_ext_grid_by_gen(ppnet, [0], slack=True)

    # Sort the tables
    ppnet.gen.sort_values("bus", ignore_index=True, inplace=True)

    # Fix certain values
    ppnet.gen.loc[3, "vm_pu"] = 1.06  # Set a value in the range (will be bypassed anyway)
    ppnet.gen.loc[4, "vm_pu"] = 1.06  # Set a value in the range (will be bypassed anyway)

    # Add labels to the generators
    ppnet.gen.loc[[0, 1], "type"] = "power generator"
    ppnet.gen.loc[[2, 3, 4], "type"] = "synchronous condenser"

    # ----- SYNCHRONOUS CONDENSERS -----
    # Set up the active power of synchronous condensers to 0
    ppnet.gen.max_p_mw.loc[[2, 3, 4]] = 0

    # ----- LOADS -----
    # Sort the loads
    ppnet.load.sort_values("bus", ignore_index=True, inplace=True)

    # ----- COORDINATES -----
    # Setup coordinates of buses
    ppnet.bus_geodata.sort_index(inplace=True)

    ppnet.bus_geodata.x = np.array(
        [1.445, 3.035, 6.785, 6.626, 4.939, 5.099, 7.776, 8.495, 6.626, 6.295, 5.485, 3.074, 4.995, 6.626])
    ppnet.bus_geodata.y = 7 - np.array(
        [3.53, 5.574, 6.636, 4.574, 4.303, 3.485, 4.012, 4.012, 3.289, 2.824, 2.633, 2.189, 1.638, 2.129])

    return ppnet


def case_normal(mm, n):
    import greyboxmodels.cpsmodels.physical.electrical.PowerFlowPowerGrid as PG
    # Get the ppnet
    ppnet = setup_ppnet()

    # Instantiate model
    pg_plant = PG.DataDrivenPowerGrid(ppnet, mm, n)

    return pg_plant


def case_exponential(mm, n):
    import greyboxmodels.cpsmodels.physical.electrical.PowerFlowPowerGrid as PG

    # Get the ppnet
    ppnet = setup_ppnet()

    # Instantiate model
    pg_plant = PG.DataDrivenPowerGrid(ppnet, mm, n)

    # Setup fail and repair rates
    # https://www.csee.org.cn/pic/u/cms/www/202103/04085233c6w4.pdf (power lines)
    fail_rate_gen = 1 / (3000 * 3600)
    # fail_rate_gen = 1 / (3600 * 24)   # For testing failure
    fail_rate_line = 1 / (3600 * 3500)
    # fail_rate_line = 1 / (3600 * 24)  # For testing failure
    fail_rate_trafo = 1 / (19000 * 3600)
    # fail_rate_trafo = 1 / (3600 * 24)  # For testing failure

    repair_rate_gen = 1 / (24 * 3600)
    # repair_rate_gen = 1 / (12 * 3600)  # For testing failure
    repair_rate_line = 1 / (12 * 3600)
    # repair_rate_line = 1 / (12 * 3600)  # For testing failure
    repair_rate_trafo = 1 / (4 * 3600)
    # repair_rate_trafo = 1 / (12 * 3600)  # For testing failure

    pg_plant.set_column_in_table("gen", "fail_rate", fail_rate_gen)
    pg_plant.set_column_in_table("gen", "repair_rate", repair_rate_gen)

    pg_plant.set_column_in_table("line", "fail_rate", fail_rate_line)
    pg_plant.set_column_in_table("line", "repair_rate", repair_rate_line)

    pg_plant.set_column_in_table("trafo", "fail_rate", fail_rate_trafo)
    pg_plant.set_column_in_table("trafo", "repair_rate", repair_rate_trafo)

    return pg_plant


# %% Run the main loop
if __name__ == '__main__':
    import torch
    import dill as pickle
    from greyboxmodels.bbmcpsmodels.cyber.feedforward_nn_opf import BBM1_SimpleNet

    # Specify locations of the plant models
    save_to1 = "data/bb-models/pg_ieee14_bbm_deterministic.pkl"
    save_to2 = "data/bb-models/pg_ieee14_bbm_exponential.pkl"

    # Specify the locations of the metamodels and normalization specs
    pf_mm_location = "D:/projects/IPTLC_BBMs/models/BBM1_SimpleNet_PF_2024-04-03_18-06-45_20240408-010659.pt"
    norm_spec_loc = "D:/projects/IPTLC_BBMs/data/training-datasets/PF/2024-04-03_18-06-45/normalization_spec.pkl"

    # Load the metamodel
    pf_mm = BBM1_SimpleNet(57, 80)
    pf_mm.load_state_dict(torch.load(pf_mm_location))
    pf_mm.eval()

    # Load the specification of the normalization
    with open(norm_spec_loc, "rb") as f:
        normalization = pickle.load(f)

    # Create the case
    plant1 = case_normal(pf_mm, normalization)
    plant2 = case_exponential(pf_mm, normalization)

    # Save the model
    plant1.save(save_to1)
    plant2.save(save_to2)
