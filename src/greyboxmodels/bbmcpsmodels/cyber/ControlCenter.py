"""
A subclass of ControlCenter that uses a metamodel to approximate the OPF solution.

Author: Juan-Pablo Futalef
"""
from greyboxmodels.cpsmodels.cyber.ControlCenter import ControlCenter
from greyboxmodels.cpsmodels.physical.electrical.PowerFlowPowerGrid import PowerGrid
from greyboxmodels.cpsmodels.cyber.TLCN import TLCNode
import torch
import numpy as np


class DataDrivenControlCenter(ControlCenter):
    def __init__(self,
                 opf_bbm,
                 power_grid: PowerGrid,
                 normalization=None,
                 init_status=1,
                 tlc_node: TLCNode.TLCNode = None,
                 ):
        super(DataDrivenControlCenter, self).__init__(power_grid, init_status, tlc_node)
        self.opf_bbm = opf_bbm

        # Set the device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Set the normalization specs
        self.normalization_spec = normalization

        # Move the model to the device
        self.opf_bbm.to(self.device)

    def optimal_power_flow(self, Pd, Qd, Vg_m, piGen, piLine, piTrafo, opf_opts=None):
        # Create the input vector
        x = np.concatenate([Pd, Qd, Vg_m, piGen, piLine, piTrafo])

        # Normalize the inputs
        x = self.normalize(x)

        # Create a tensor of the controlled inputs
        input_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)

        # Run the model
        y = self.opf_bbm(input_tensor.float())

        # Denormalize the output
        y = y.cpu().detach().numpy()
        y = self.denormalize(y)

        return y

    def normalize(self, x):
        if self.normalization_spec is None:
            return x

        type = self.normalization_spec["type"]

        if type == "minmax":
            return self.minmax_normalize(x)

    def denormalize(self, x):
        if self.normalization_spec is None:
            return x

        type = self.normalization_spec["type"]

        if type == "minmax":
            return self.minmax_denormalize(x)

    def minmax_normalize(self, x):
        x_min = self.normalization_spec["input_min"]
        x_max = self.normalization_spec["input_max"]
        return (x - x_min) / (x_max - x_min)

    def minmax_denormalize(self, x):
        x_min = self.normalization_spec["output_min"]
        x_max = self.normalization_spec["output_max"]
        return x * (x_max - x_min) + x_min
