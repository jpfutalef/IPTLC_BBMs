import os
from pathlib import Path
from importlib import reload

import torch

import greyboxmodels.bbmcpsmodels.nn_creator as creator
import greyboxmodels.bbmcpsmodels.physical.feedforward_nn_pf as pf_bbm

# Set the working directory
print(f"Working directory: {os.getcwd()}")

# Check GPU availability
device = creator.get_device()
print(f"Device: {device}")


#%% Specify the paths
# datasets_folder = Path("sim_data/IO-datasets/PF/2024-03-20_18-55-20")
datasets_folder = Path("sim_data/IO-datasets/PF/2024-04-03_18-06-45")
model_folder = Path("models/PF/").resolve()

print(f"Datasets folder: {datasets_folder}")
print(f"Model folder: {model_folder}")


#%% Set up the BBM creator]
BBM_creator = creator.BBMCreator()

# Set up the datasets
dataset_name = f"{datasets_folder.parent.name}_{datasets_folder.name}"

# Load the datasets
try:
    print(f"Loading dataloaders from {model_folder}")
    with open(model_folder / f"dataloaders_{dataset_name}.pth", "rb") as f:
        loaders = torch.load(f)
        print("        Dataloaders loaded successfully!")

except FileNotFoundError:
    # Create the dataloaders
    print(f"        Not found... creating dataloaders for {dataset_name}")
    loaders = creator.setup_datasets(datasets_folder,
                                     dataset_name,
                                     remove_nans=True,
                                     ratios=(0.70, 0.15, 0.15),
                                     batch_size=32,
                                     input_name="input_normalized.npy",
                                     output_name="output_normalized.npy")

    # Save the dataloaders
    with open(model_folder / f"dataloaders_{dataset_name}.pth", "wb") as f:
        torch.save(loaders, f)
        print("        Dataloaders saved successfully!")

BBM_creator.set_dataloaders(*loaders)


#%% BBM 1: a two-layer feedforward neural network
BBM_creator.instantiate_model(pf_bbm.BBM1_SimpleNet, 57, 80)


# Train the model
BBM_creator.train(save_to=model_folder, epochs=100)
BBM_creator._summary()


#%% BBM 2: a two-layer feedforward neural network
BBM_creator.instantiate_model(pf_bbm.BBM2_DeepNN, 57, 80)

# Train the model
BBM_creator.train(save_to=model_folder, epochs=100)
BBM_creator._summary()

#%% BBM 3: a two-layer feedforward neural network
BBM_creator.instantiate_model(pf_bbm.BBM3)

# Train the model
BBM_creator.train(save_to=model_folder, epochs=100)
BBM_creator._summary()

#%% BBM 4: a two-layer feedforward neural network
BBM_creator.instantiate_model(pf_bbm.BBM4)

# Train the model
BBM_creator.train(save_to=model_folder, epochs=100)
BBM_creator._summary()

