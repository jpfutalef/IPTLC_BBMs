"""
This script creates a data-driven control center for the IEEE14 power grid.

The power grid object should have already been created and saved in a file since we use it to create the control center.

Author: Juan-Pablo Futalef
"""


# %% Create the objects
def case(pg_plant_path, metamodel, normalization):
    """
    Creates a control center for the IEEE14 power grid.
    :param pg_plant_path: the path to the power grid plant
    @return: the control center
    """
    from greyboxmodels.cpsmodels.Plant import Plant
    import greyboxmodels.cpsmodels.cyber.ControlCenter as CC

    # Read the power grid plant
    pg_plant = Plant.load(pg_plant_path)

    # Get the ppnet from the power grid plant
    ppnet = pg_plant.ppnet

    # ----- OPF COST FUNCTION -----
    # Delete all the rows in poly_cost
    # pg.poly_cost.drop(pg.poly_cost.index, inplace=True)

    # Cost function for the generators
    # Check: https://www.mercatoelettrico.org/it/
    # Check: https://www.pmstudiotecnico.it/nuove-aliquote-per-lenergia-reattiva-2023/#:~:text=per%20energia%20reattiva%20oltre%20il%2075%25%20della%20attiva%20(F1%2C,1%2C689%20centesimi%20di%20euro%2FkVarh
    # We consider that the larger generator produces active power 10% cheaper than the smaller one.
    # pp.create_poly_cost(pg, 0, 'gen',
    #                     cp0_eur=0,
    #                     cp1_eur_per_mw=70. * .9,
    #                     cp2_eur_per_mw2=.05 * .9,
    #                     cq0_eur=0,
    #                     cq1_eur_per_mvar=1.5,
    #                     cq2_eur_per_mvar2=0.01)
    #
    # pp.create_poly_cost(pg, 1, 'gen',
    #                     cp0_eur=0,
    #                     cp1_eur_per_mw=70.,
    #                     cp2_eur_per_mw2=.05,
    #                     cq0_eur=0,
    #                     cq1_eur_per_mvar=1.5,
    #                     cq2_eur_per_mvar2=0.01)
    #
    # # Now, for synchronous condensers. We add a fix cost for cp0, and 100 for cp1 to ensure that the cost is always
    # # higher than the generators (active power). For the reactive power, we consider 90% of that of the generators.
    # pp.create_poly_cost(pg, 2, 'gen',
    #                     cp0_eur=1000,
    #                     cp1_eur_per_mw=100.,
    #                     cp2_eur_per_mw2=0.,
    #                     cq0_eur=0,
    #                     cq1_eur_per_mvar=1.5 * .9,
    #                     cq2_eur_per_mvar2=0.01 * .9)
    #
    # pp.create_poly_cost(pg, 3, 'gen',
    #                     cp0_eur=1000,
    #                     cp1_eur_per_mw=100.,
    #                     cp2_eur_per_mw2=0.,
    #                     cq0_eur=0,
    #                     cq1_eur_per_mvar=1.5 * .9,
    #                     cq2_eur_per_mvar2=0.01 * .9)
    #
    # pp.create_poly_cost(pg, 4, 'gen',
    #                     cp0_eur=1000,
    #                     cp1_eur_per_mw=100.,
    #                     cp2_eur_per_mw2=0.,
    #                     cq0_eur=0,
    #                     cq1_eur_per_mvar=1.5 * .9,
    #                     cq2_eur_per_mvar2=0.01 * .9)

    # --- CONTROL CENTER ---
    cc = CC.DataDrivenControlCenter(pg_plant, metamodel, normalization)

    return cc


# %% Run the main loop
if __name__ == '__main__':
    import torch
    import dill as pickle
    from greyboxmodels.bbmcpsmodels.cyber.feedforward_nn_opf import BBM1_SimpleNet

    # Specify locations of the plant models
    pg_loc = "data/bb-models/pg_ieee14_bbm_deterministic.pkl"

    # Location of the power grid case
    save_to = "data/bb-models/cc_ieee14_bbm.pkl"

    # Specify the locations of the metamodels and normalization specs
    mm_loc = "D:/projects/IPTLC_BBMs/models/OPF/BBM1_SimpleNet_OPF_2024-04-03_18-06-45_20240408-135438.pt"
    norm_spec_loc = "D:/projects/IPTLC_BBMs/data/training-datasets_development/OPF/2024-04-03_18-06-45/normalization_spec.pkl"

    # Load the metamodel
    mm = BBM1_SimpleNet(52, 10)
    mm.load_state_dict(torch.load(mm_loc))
    mm.eval()

    # Load the specification of the normalization
    with open(norm_spec_loc, "rb") as f:
        normalization = pickle.load(f)

    # Create the case
    plant = case(pg_loc, mm, normalization)

    # Save the model
    plant.save(save_to)
