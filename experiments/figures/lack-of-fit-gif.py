"""
This script creates an animated GIF for the exemplification of the calciulation of a dynamic lack of fit in time.

The idea of the gif is to show how how the KS statistic evolves in time using two different distributions.

The original signal is a sine wave with a frequency of 1 Hz and an amplitude of 1 contaminated with a white noise with a standard deviation of 0.1.

The proposed signal is a sine wave with a frequency of 1 Hz and an amplitude of 1 contaminated with a white noise with a standard deviation of 0.4.

Author: Juan-Pablo Futalef
"""
import numpy as np
import matplotlib.pyplot as plt

from greyboxmodels.voi.metrics import lack_of_fit as lof

#%% Create the data
time = np.linspace(0, 2, 20)
signal = 2 * np.sin(2 * np.pi * time)

n_realizations = 100

# Original
noise = np.random.normal(0, 0.1, size=(signal.shape[0], n_realizations))
original = np.tile(signal, (n_realizations, 1)).T + noise

# Proposed
noise = np.random.normal(0, 0.4, size=(signal.shape[0], n_realizations))
proposed = np.tile(signal, (n_realizations, 1)).T + noise

#%% Create the figure of both signals
fig, ax = plt.subplots()

# Create the lines
for i in range(n_realizations):
    ax.plot(time, original[:, i], '-b', alpha=0.2)
    ax.plot(time, proposed[:, i], '-r', alpha=0.2)

ax.set_xlim([0, time[-1]])

plt.show()

#%% Lack of fit
lof_array = []
info_array = []
for i in range(len(time)):
    lof_val, info = lof.lack_of_fit(original[i, :], proposed[i, :], n_bins=50, return_info=True)
    lof_array.append(lof_val)
    info_array.append(info)

lof_array = np.array(lof_array)

#%% Plot
fig, ax = plt.subplots()

ax.plot(time, lof_array, '-k')

plt.show()

#%% A cdf comparison plot
fig, ax = plt.subplots()

idx = 0
ax.plot(info_array[idx]["bins"], info_array[idx]["ecdf_1"], '-b')
ax.plot(info_array[idx]["bins"], info_array[idx]["ecdf_2"], '-r')

plt.show()

