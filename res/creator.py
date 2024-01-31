# Import the necessary libraries
import os
import datetime
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from torch.utils.data import TensorDataset


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


# A general class to store the settings of the model and the training process
class BBMCreator:
    def __init__(self,
                 training_dataloader,
                 validation_dataloader=None,
                 test_dataloader=None,
                 criterion=None,
                 optimizer=None,
                 epochs=1000,
                 print_every=100,
                 device=None,
                 cross_validation=False
                 ):
        self.model = None
        self.optimizer = optimizer
        self.train_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.test_dataloader = test_dataloader
        self.criterion = criterion if criterion is not None else nn.MSELoss()
        self.epochs = epochs
        self.print_every = print_every
        self.device = device if device is not None else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.init_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.best_val = np.inf

    def instantiate_model(self, model_class, *args, **kwargs):
        self.model = model_class(*args, **kwargs)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.best_val = np.inf

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.best_val = np.inf

    def train_epoch(self, epoch_number):
        # Setup a tqdm progress bar
        bar = tqdm.tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader),
                        desc=f"Epoch {epoch_number + 1:4}/{self.epochs}",
                        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}")

        running_loss = 0.0
        total_batches = 0
        for i, (inputs, benchmark_outputs) in bar:
            # Move the data to the device
            inputs = inputs.to(self.device)
            benchmark_outputs = benchmark_outputs.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forwards prediction
            predicted_outputs = self.model(inputs)

            # Compute the loss and back-propagate
            loss = self.criterion(predicted_outputs, benchmark_outputs)
            loss.backward()

            # Optimize
            self.optimizer.step()

            # Add the loss to the running loss
            running_loss += loss.item()

            # Update the total number of batches
            total_batches += 1

            # Update the progress bar every 100 batches
            if (i + 1) % 100 == 0:
                # Get the best val string: if the best val is inf, then print N/A; if it's larger than 0.01, then print 2 decimals; otherwise, print in scientific notation
                best_val_str = "N/A" if self.best_val == np.inf else f"{self.best_val:.2f}" if self.best_val > 0.01 else f"{self.best_val:.2e}"
                bar.set_description(
                    f"Epoch {epoch_number + 1}/{self.epochs} (Running loss: {running_loss / total_batches:.2f}, Avrg. batch loss: {running_loss / total_batches:.2e}, Best val. loss: {best_val_str})")

            # Print inputs, outputs, and loss
            if torch.isnan(loss):
                print(f"Epoch {epoch_number + 1}, Batch {i + 1}")
                print("Inputs: ", inputs)
                print("Outputs: ", predicted_outputs)
                print("Loss: ", loss)

                # Stop the training
                raise Exception("NaN loss")

    def train(self, save_to: Path = None, t_cowntdown=3, **kwargs):
        # Indicate the name of the model that will be trained
        print(f"Training model '{self.model.name}'")

        # Modify this instance with the keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Before training, check if the save_to path exists
        if save_to is not None:
            print(f"The folder '{save_to}/' will be created if it does not exist")
            os.makedirs(save_to, exist_ok=True)

        # Print device used for training
        print(f"Training on {self.device}")

        # Initialize countdown
        countdown(t_cowntdown, message="Training starts in: ")

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
                        self.save_model(save_to, include_timestamp=False, extra_name=f"_{self.init_timestamp}_best")

        # Print a summary of the performance of the model
        print(f"Train loss: {self.evaluate_loss(*next(iter(self.train_dataloader))):.2f}")
        if self.validation_dataloader is not None:
            print(f"Validation loss: {self.evaluate_loss(*next(iter(self.validation_dataloader))):.2f}")
        if self.test_dataloader is not None:
            print(f"Test loss: {self.evaluate_loss(*next(iter(self.test_dataloader))):.2f}")

        # Print a message indicating that the training has finished
        print("Finished!")

        # Save the model
        if save_to is not None:
            self.save_model(save_to)

    def save_model(self, save_to: Path, include_timestamp=True, extra_name=""):
        save_to = Path(save_to)
        model_name = self.model.name
        model_name += extra_name
        if include_timestamp:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_name += f"_{timestamp}"
        torch.save(self.model.state_dict(), save_to / Path(f"{model_name}.pt"))

    def evaluate_loss(self, X, Y):
        # Check if data is in the device, and move it if not
        X = X.to(self.device)
        Y = Y.to(self.device)

        # Compute the outputs and the loss
        predicted_Y = self.model(X)
        loss = self.criterion(predicted_Y, Y).item() * len(X)

        # Return the average loss
        return loss / len(X)

    def evaluate_loss_dataloader(self, dataloader):
        # Compute the loss for the entire dataset
        loss = 0.0
        for X, Y in dataloader:
            # Compute the loss for the batch
            loss += self.evaluate_loss(X, Y)

        # Return the average loss
        return loss / len(dataloader)

    def summary(self, save_to: Path = None):
        """
        Creates a table with the summary of the performance of the model
        :return:
        """
        # Check if a results table is already saved
        if save_to is not None:
            save_to = Path(save_to, "model_performances.csv")
            if save_to.exists():
                # If it exists, load it and add the new results
                df = pd.read_csv(save_to, index_col=0)
                df = df.append(self._summary(), ignore_index=True)
            else:
                # If it does not exist, create a new one
                df = self._summary()

            # Save the dataframe
            df.to_csv(save_to)
        else:
            # If no path is provided, just print the summary
            df = self._summary()

        # Return the dataframe
        return df

    def _summary(self):
        """
        Creates a table with the summary of the performance of the model
        :return:
        """
        # Create a dataframe to store the results
        df = pd.DataFrame(
            columns=["Model", "Train loss", "Validation loss", "Test loss", "Input dataset", "Output dataset",
                     "Input size", "Output size"])

        # Compute the losses
        train_loss = self.evaluate_loss(*next(iter(self.train_dataloader)))
        if self.validation_dataloader is not None:
            val_loss = self.evaluate_loss(*next(iter(self.validation_dataloader)))
        else:
            val_loss = np.nan
        if self.test_dataloader is not None:
            test_loss = self.evaluate_loss(*next(iter(self.test_dataloader)))
        else:
            test_loss = np.nan

        # Dataset names
        input_dataset = self.train_dataloader.dataset.name

        # Add the results to the dataframe
        df.loc[0] = [self.model.name, train_loss, val_loss, test_loss]

        # Return the dataframe
        return df
