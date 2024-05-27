"""
This script creates objects that can be written to a file and later read by the simulator to run the simulation.

Those objects should be used in all simulations for consistency.

Author: Juan-Pablo Futalef
"""
from pathlib import Path

import greyboxmodels.cpsmodels.cyber.ControlCenter as CC
import greyboxmodels.cpsmodels.physical.electrical.PowerFlowPowerGrid as PG

#%% Specify locations of the files
bbm_dir = Path("data/gbm-model-creation/control_center_bbm.pkl")

#%% Create the objects
pg = PG.PowerGridExponential()
control_center = CC.DataDrivenControlCenter()

