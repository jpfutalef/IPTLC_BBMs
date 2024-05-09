# Import the necessary libraries
import os
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


def countdown(t, message=None):
    import time
    while t >= 0:
        mins, secs = divmod(t, 60)
        timeformat = '{:02d}:{:02d}'.format(mins, secs)
        if message is not None:
            print(message, end=' ')
        print(timeformat, end='\r')
        time.sleep(1)
        t -= 1


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def setup_datasets(dataset_folder,
                   dataset_name,
                   remove_nans=True,
                   ratios=(0.70, 0.15, 0.15),
                   batch_size=32,
                   input_name="input.npy",
                   output_name="output.npy",
                   dataset_name_is_folder=False):
    if dataset_name_is_folder:
        dataset_folder = dataset_folder / dataset_name
    dataset_metadata = DatasetMetadata(name=dataset_name)

    # Print the dataset name
    print(f"Loading: {dataset_folder}")

    # Load the sim_data
    X_np = np.load(dataset_folder / input_name)
    Y_np = np.load(dataset_folder / output_name)

    print("---- Dataset loaded ----")
    print(f"    Input shape: {X_np.shape}")
    print(f"    Output shape: {Y_np.shape}")

    # Remove NaNs
    if remove_nans:
        print("---- Removing NaNs ----")
        # Locate the rows with NaNs
        X_nan_rows = np.where(np.isnan(X_np))[0]
        Y_nan_rows = np.where(np.isnan(Y_np))[0]

        # Delete in X and Y the rows with NaNs
        rows_to_delete = np.union1d(X_nan_rows, Y_nan_rows)
        print(f"    Rows to delete: {rows_to_delete}")

        X_np = np.delete(X_np, rows_to_delete, axis=0)
        Y_np = np.delete(Y_np, rows_to_delete, axis=0)

        print(f"    Input shape: {X_np.shape}")
        print(f"    Output shape: {Y_np.shape}")

    # Convert to torch tensors
    print("---- Converting to torch tensors ----")
    X = torch.from_numpy(X_np).float()
    Y = torch.from_numpy(Y_np).float()

    # Create the dataset
    dataset = CustomTensorDataset(X, Y, metadata=dataset_metadata)  # dataset[i] = (X[i], Y[i])

    # Split the dataset
    training_dataset, validation_dataset, test_dataset = dataset.split(*ratios)

    # Create the dataloaders
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Print end of loading
    print("---- Dataset loaded! ----")

    # Return the dataloaders
    return training_dataloader, validation_dataloader, test_dataloader


def open_example_dfs(datasets_folder, dataset_name, n_rows=20):
    dataset_folder = datasets_folder / dataset_name

    input_df = pd.read_csv(dataset_folder / "inputs.csv", index_col=0, header=0, nrows=n_rows)
    output_df = pd.read_csv(dataset_folder / "outputs.csv", index_col=0, header=0, nrows=n_rows)

    single_input_df = pd.read_csv(dataset_folder / "single_input.csv", index_col=0, header=0, nrows=n_rows)
    single_output_df = pd.read_csv(dataset_folder / "single_output.csv", index_col=0, header=0, nrows=n_rows)

    return input_df, output_df, single_input_df, single_output_df


# A metadata dataclass
@dataclass
class DatasetMetadata:
    name: str
    description: str = None
    input_names: list = None
    output_names: list = None
    input_units: list = None
    output_units: list = None


# A custom TensorDataset class to include metadata
class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of metadata.
    """

    def __init__(self, *tensors, metadata: DatasetMetadata = None):
        super(CustomTensorDataset, self).__init__(*tensors)
        self.metadata = metadata

    def split(self, training_ratio=0.8, validation_ratio=0.1, test_ratio=0.1):
        # Assert that the ratios sum to 1
        sum_ratios = training_ratio + validation_ratio + test_ratio
        assert sum_ratios == 1, f"The ratios must sum to 1, but they sum to {sum_ratios}"

        # Get the number of samples
        n_samples = len(self)

        # Compute the number of samples for each set
        train_size = int(n_samples * training_ratio)
        validation_size = int(n_samples * validation_ratio)
        test_size = n_samples - train_size - validation_size

        # Split the dataset
        result = torch.utils.data.random_split(self, [train_size, validation_size, test_size])

        # Unpack the result
        training_dataset, validation_dataset, test_dataset = result

        # Return the datasets
        return training_dataset, validation_dataset, test_dataset


# A general class to store the settings of the model and the training process
class BBMCreator:
    def __init__(self,
                 training_dataloader=None,
                 validation_dataloader=None,
                 test_dataloader=None,
                 criterion=None,
                 optimizer=None,
                 epochs=1000,
                 print_every=100,
                 device=None,
                 cross_validation=False,
                 print_summary_at_the_end=True
                 ):
        self.train_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.optimizer = optimizer
        self.epochs = epochs
        self.print_every = print_every
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cross_validation = cross_validation
        self.print_summary_at_the_end = print_summary_at_the_end

        # These variables will be set at the beginning of the training
        self.model = None
        self.best_val = None
        self.training_time = None
        self.init_timestamp = None
        self.save_to = None
        self.save_model_to = None

    def setup_training_vars(self, save_to):
        self.best_val = np.inf
        self.training_time = None
        self.init_timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.save_to = Path(save_to) if save_to is not None else None
        self.save_model_to = None

    def instantiate_model(self, model_class, *args, **kwargs):
        self.model = model_class(*args, **kwargs)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def set_dataloaders(self, training_dataloader, validation_dataloader, test_dataloader):
        self.train_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader

    def train_epoch(self, epoch_number):
        # Setup a tqdm progress bar
        bar = tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),
                        desc=f"Epoch {epoch_number + 1:4}/{self.epochs}",
                        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        running_loss = 0.0
        total_batches = 0
        for i, (inputs, benchmark_outputs) in bar:
            # Move the sim_data to the device
            inputs = inputs.to(self.device)
            benchmark_outputs = benchmark_outputs.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forwards prediction
            predicted_outputs = self.model(inputs)

            # Compute the loss and back-propagate
            loss = self.criterion(predicted_outputs, benchmark_outputs) # I assume the loss is averaged over the batch
            loss.backward()

            # Optimize
            self.optimizer.step()

            # Add the loss to the running loss
            running_loss += loss.item()

            # Update the total number of batches
            total_batches += 1

            # Update the progress bar every 100 batches
            if (i + 1) % 100 == 0:
                # Get the val string: if val is inf, then print N/A; if it's larger than 0.01, then print 2 decimals; otherwise, print in scientific notation
                best_val_str = "N/A" if self.best_val == np.inf else f"{self.best_val:.2f}" if self.best_val > 0.01 else f"{self.best_val:.2e}"

                bar.set_description(
                    f"Epoch {epoch_number + 1}/{self.epochs} (Loss - Train: {running_loss / total_batches:.2e}, Best val: {best_val_str})")

            # Print inputs, outputs, and loss
            if torch.isnan(loss):
                print(f"Epoch {epoch_number + 1}, Batch {i + 1}")
                print("Inputs: ", inputs)
                print("Outputs: ", predicted_outputs)
                print("Loss: ", loss)

                # Stop the training
                raise Exception("NaN loss")

    def train(self, save_to: Path = None, t_cowntdown=3, **kwargs):
        # Check that the dataloaders are not None
        assert self.train_dataloader is not None, "The training dataloader is None"
        assert self.validation_dataloader is not None, "The validation dataloader is None"
        assert self.test_dataloader is not None, "The test dataloader is None"

        # Indicate the name of the model that will be trained
        print(f"------ Training model '{self.model.name}' ------")

        # Setup the training variables
        self.setup_training_vars(save_to)

        # Modify this instance with the keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Before training, check if the save_to path exists
        if save_to is not None:
            print(f"Models and summary will be saved to '{save_to}' (will be created if it does not exist)")
            print(f"    - Model path: {self._develop_model_path()}")
            print(f"    - Summary path: {self.save_to / 'models_summary.csv'}")
            os.makedirs(save_to, exist_ok=True)

        # Print device used for training
        print(f"Training on {self.device}")

        # Initialize countdown
        countdown(t_cowntdown, message="Training starts in: ")

        # Initialize time counter
        start_time = datetime.datetime.now()

        # Training loop
        for epoch_number in range(self.epochs):
            # Train the model for one epoch
            self.train_epoch(epoch_number)

            # Evaluate the model on the validation set
            if self.validation_dataloader is not None:
                val_loss = self.evaluate_loss_dataloader(self.validation_dataloader)
                if val_loss < self.best_val:
                    self.best_val = val_loss
                    if save_to is not None:
                        self.save_model(extra_name=f"_best")

        # Compute the training time in milliseconds
        training_time = datetime.datetime.now() - start_time
        self.training_time = training_time.total_seconds() * 1000

        # Print a summary of the performance of the model
        self.summary(print_output=self.print_summary_at_the_end)

        # Print a message indicating that the training has finished
        print("------ Finished! ------")

        # Save the model
        self.save_model()

    def save_model(self, extra_name=""):
        # Check if a path was provided
        if self.save_to is None:
            return

        # Develop the model path
        path = self._develop_model_path()

        # Add extra name
        path = path.parent / Path(f"{path.stem}{extra_name}{path.suffix}")

        # Save the model
        torch.save(self.model.state_dict(), str(path))

    def _develop_model_path(self) -> Union[Path, None]:
        # Check if the model path has already been developed
        if self.save_model_to is not None:
            return self.save_model_to

        # Check if the save_to path exists
        if self.save_to is None:
            return None

        # Create the name of the file
        model_name = self.model.name
        dataset_name = self.train_dataloader.dataset.dataset.metadata.name
        timestamp = self.init_timestamp

        # Develop the model path
        path = self.save_to / Path(f"{model_name}_{dataset_name}_{timestamp}.pt")

        return path

    def evaluate_loss(self, X, Y):
        # Check if sim_data is in the device, and move it if not
        X = X.to(self.device)
        Y = Y.to(self.device)

        # Compute the outputs and the loss
        predicted_Y = self.model(X)
        loss = self.criterion(predicted_Y, Y).item()    # I assume the loss is averaged over the batch

        # Return the average loss
        return loss

    def evaluate_loss_dataloader(self, dataloader):
        # Compute the loss for the entire dataset
        loss = 0.0
        n_batch = 0
        for X, Y in dataloader:
            # Compute the average loss for the batch
            batch_loss = self.evaluate_loss(X, Y)

            # Add the batch loss to the total loss
            loss += batch_loss

            # Update the number of batches
            n_batch += 1

        # Return the average loss
        return loss / n_batch

    def summary(self, print_output=True):
        """
        Creates a table with the summary of the performance of the model
        :return:
        """
        # Get the summary of this model
        df = self._summary()

        # Print the dataframe
        if print_output:
            print(df)

        # Check if the results table should be saved
        if self.save_to is not None:
            # Append the dataframe to the results table
            df = self._append_to_results_table(df)

        # Return the dataframe
        return df

    def _append_to_results_table(self, df):
        # Check if the results table exists
        results_table_path = self.save_to / "models_summary.csv"

        if not results_table_path.exists():
            # The results table does not exist, so create it
            df.to_csv(results_table_path)
            return df

        # Load the results table
        results_table = pd.read_csv(results_table_path, index_col=[0, 1, 2])

        # Check if the model is already in the results table; if so, return the results table
        model_name = self.model.name
        dataset_name = self.train_dataloader.dataset.dataset.metadata.name
        timestamp = self.init_timestamp
        comparison = (model_name, dataset_name, timestamp) in results_table.index
        if comparison:
            return df

        # Concatenate the results table with the new results
        results_table = pd.concat([results_table, df])

        # Save the results table
        results_table.to_csv(results_table_path)

        # Return the results table
        return results_table

    def _summary(self):
        """
        Creates a table with the summary of the performance of the model
        :return:
        """
        # Model name
        model_name = self.model.name

        # Dataset name
        dataset_name = self.train_dataloader.dataset.dataset.metadata.name

        # Timestamp
        timestamp = self.init_timestamp

        # Compute the losses
        train_loss = self.evaluate_loss_dataloader(self.train_dataloader)
        val_loss = self.evaluate_loss_dataloader(self.validation_dataloader)
        test_loss = self.evaluate_loss_dataloader(self.test_dataloader)

        # Input and output sizes
        input_size = self.train_dataloader.dataset.dataset.tensors[0].shape[1]
        output_size = self.train_dataloader.dataset.dataset.tensors[1].shape[1]

        # Training time
        training_time = self.training_time

        # Create a dictionary with the results
        results = {"Model": model_name,
                   "Dataset": dataset_name,
                   "Timestamp": timestamp,
                   "Train loss": train_loss,
                   "Validation loss": val_loss,
                   "Test loss": test_loss,
                   "Input size": input_size,
                   "Output size": output_size,
                   "Training time [ms]": training_time,
                   "Model path": str(self._develop_model_path())
                   }

        # Create a dataframe with the results
        df = pd.DataFrame(results, index=[0])

        # Create MultiIndex setting the following columns as the index: (Model, Dataset, Timestamp)
        df.set_index(["Model", "Dataset", "Timestamp"], inplace=True)

        # Return the dataframe
        return df
