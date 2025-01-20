import os
from pathlib import Path

from greyboxmodels.voi.metrics import computational_load as cl


# %% Specify paths
wbm = Path("D:/projects/CPS-SenarioGeneration/data/iptlc/MonteCarlo/2024-05-09_15-16-30")
gbm1 = Path("data/gbm-simulations/iptlc/arch_1-0_0_1/2024-05-09_15-16-30")
gbm2 = Path("data/gbm-simulations/iptlc/arch_2-1_0_0/2024-05-09_15-16-30")
gbm3 = Path("data/gbm-simulations/iptlc/arch_3-1_0_1/2024-05-09_15-16-30")
gbm4 = Path("data/gbm-simulations/iptlc/arch_4-0_1_0/2024-05-09_15-16-30")
gbm5 = Path("data/gbm-simulations/iptlc/arch_5-0_1_1/2024-05-09_15-16-30")
gbm6 = Path("data/gbm-simulations/iptlc/arch_6-1_1_0/2024-05-09_15-16-30")
gbm7 = Path("data/gbm-simulations/iptlc/arch_7-1_1_1/2024-05-09_15-16-30")

# %% Add to list
folders = [wbm, gbm1, gbm2, gbm3, gbm4, gbm5, gbm6, gbm7]

#%% Create a folder to store the values
os.makedirs(Path("data/voi_losses") / "l1" / wbm.name, exist_ok=True)

# %% Compute
for folder in folders:
    print(f"    ------- Computing for {folder} -------")
    comp_load, info = cl.computational_load_folder(folder)

    # Save the results
    comp_load.to_csv(folder / "comp_load.csv")


#%% main test
if __name__ == '__main__':
    import dill as pickle
    target = wbm / "simulation_0.pkl"

    with open(target, "rb") as f:
        sim_data = pickle.load(f)

    target2 = gbm1 / "simulation_0.pkl"
    with open(target2, "rb") as f:
        sim_data2 = pickle.load(f)
