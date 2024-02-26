"""
A subclass of ControlCenter that uses a metamodel to approximate the OPF solution.

Author: Juan-Pablo Futalef
"""

from greyboxmodels.cpsmodels.cyber.ControlCenter import ControlCenter
import torch


class DataDrivenControlCenter(ControlCenter):
    def __init__(self, power_grid,
                 opf_bbm,
                 init_status=1,
                 torch_device=None):
        super(DataDrivenControlCenter, self).__init__(power_grid, init_status)
        self.opf_bbm = opf_bbm

        # Set the device
        if torch_device is None:
            self.device = torch.device('gpu')
        else:
            self.device = torch_device

    def optimal_power_flow(self, Pd, Qd, Vg_m, Vg_a, piGen, piLine, piTrafo, opf_opts=None):
        # Create a tensor of the controlled inputs
        input_tensor = torch.tensor([Pd, Qd, Vg_m, Vg_a, piGen, piLine, piTrafo, ])

        # Run the model
        output = self.opf_bbm(input_tensor)

        return x
