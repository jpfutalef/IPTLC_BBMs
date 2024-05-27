def case(pg_loc, cc_loc, tlc_loc):
    from greyboxmodels.cpsmodels.Plant import Plant
    from greyboxmodels.cpsmodels.cyberphysical.IPTLC.IPTLC import IPandTLC

    # load the models
    pg = Plant.load(pg_loc)
    cc = Plant.load(cc_loc)
    tlc = Plant.load(tlc_loc)

    # Set up the tlcn nodes in the control center
    bus_to_tlc_node_map = {0: 0,
                           1: 1,
                           2: 2,
                           3: 3,
                           4: 4,
                           5: 4,
                           6: 0,
                           7: 3,
                           8: 3,
                           9: 5,
                           10: 5,
                           11: 0,
                           12: 5,
                           13: 5,
                           }
    cc_tlc_node_id = 6

    # Create the case
    iptlcn = IPandTLC(pg, tlc, cc, bus_to_tlc_node_map, cc_tlc_node_id)

    return iptlcn


if __name__ == '__main__':
    # Specify locations of the files
    pg_loc = "D:/projects/Hierarchical_CPS_models/data/wbm-models/pg_ieee14_wbm_deterministic.pkl"
    tlc_loc = "D:/projects/Hierarchical_CPS_models/data/wbm-models/tlcn_case7_wbm.pkl"
    cc_loc = "D:/projects/IPTLC_BBMs/data/bb-models/cc_ieee14_bbm.pkl"

    save_to = "data/gb-models/iptlc/arch_1-0_0_1.pkl"

    # Create the cc
    iptlc = case(pg_loc, cc_loc, tlc_loc)

    # Save the model
    iptlc.save(save_to)
