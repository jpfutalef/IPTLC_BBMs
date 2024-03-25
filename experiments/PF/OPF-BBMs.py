import os
from pathlib import Path
from importlib import reload

import greyboxmodels.bbmcpsmodels.nn_creator as creator
import greyboxmodels.bbmcpsmodels.cyber.feedforward_nn_opf as opf_bbm

# Set the working directory
print(f"Working directory: {os.getcwd()}")

# Check GPU availability
device = creator.get_device()
print(f"Device: {device}")


#%% Specify the paths
datasets_folder = Path("data/IO-datasets/OPF/2024-03-20_18-55-20")


#%% Set up the BBM creator]
reload(creator)
BBM_creator = creator.BBMCreator()

# Set up the datasets
dataset_name = "MinMaxNormalizedOPF"
loaders = creator.setup_datasets(datasets_folder,
                                 dataset_name,
                                 remove_nans=True,
                                 ratios=(0.70, 0.15, 0.15),
                                 batch_size=32,
                                 input_name="opf_inputs_minmax_normalized.npy",
                                 output_name="opf_outputs_minmax_normalized.npy")
BBM_creator.set_dataloaders(*loaders)


#%% BBM 1: a two-layer feedforward neural network
input_size = loaders[0].dataset[0][0].shape[0]
output_size = loaders[0].dataset[0][1].shape[0]

BBM_creator.instantiate_model(opf_bbm.BBM1_SimpleNet, input_size, output_size)


# Train the model
BBM_creator.train(save_to=Path("models"), epochs=100)
BBM_creator._summary()


#%% BBM 2: a two-layer feedforward neural network
input_size = loaders[0].dataset[0][0].shape[0]
output_size = loaders[0].dataset[0][1].shape[0]

BBM_creator.instantiate_model(opf_bbm.BBM2_DeepNN, input_size, output_size)

# Train the model
BBM_creator.train(save_to=Path("models"), epochs=100)
BBM_creator._summary()



