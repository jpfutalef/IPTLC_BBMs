def case(pg_loc, cc_loc):
    from greyboxmodels.cpsmodels.Plant import Plant
    import greyboxmodels.cpsmodels.cyberphysical.ControlledPowerGrid.ControlledPowerGrid as CPG

    # load the models
    pg = Plant.load(pg_loc)
    cc = Plant.load(cc_loc)

    # Create the case
    cpg = CPG.ControlledPowerGrid(pg, cc)

    return cpg


if __name__ == '__main__':
    # Specify locations of the files
    pg_loc = "D:/projects/Hierarchical_CPS_models/data/wbm-models/pg_ieee14_wbm_deterministic.pkl"
    cc_loc = "D:/projects/IPTLC_BBMs/data/bb-models/cc_ieee14_bbm.pkl"
    cpg_loc = "D:/projects/IPTLC_BBMs/data/gb-models/cpg/arch1_0-1.pkl"

    # Create the cc
    cpg = case(pg_loc, cc_loc)

    # Save the model
    cpg.save(cpg_loc)
