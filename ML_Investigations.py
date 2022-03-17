# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 00:46:54 2022

@author: user
"""

#%%
import Machine_Learning as ml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.constants import constants
from matplotlib.colors import LogNorm
import Median_Filter as mf

#%%

folders = ["Dataset_2", "Dataset_3"]
#ml.unzip(foldername)
pixel_map_32, pixel_map_64, res = [], [], []
for folder in folders:
    dataset = ml.getData(folder, ["Thermal energy of beam (MeV)", 'Number of accepted particles in beam', "Maximum energy of beam (MeV)"])
    pixel_map_32 += dataset[0]
    pixel_map_64 += dataset[1]
    res += dataset[3]


#%% Noise addition
x_range_mm = [0, 35]
y_range_mm = [0, 35]

weight = 0.1
background_noise = 10
abberations = 0.2
hard_hit_noise_fraction = 0.1
detector_threshold = 255
zero_point_micron = 200
multiplier = 1

noisy_data_32, res = ml.add_noise(pixel_map_32, res, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
clean_data_32, _ = ml.add_noise(pixel_map_32, res, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
# Initial Processing of data
noisy_processed_32 = ml.preprocess(noisy_data_32)
clean_processed_32 = ml.preprocess(clean_data_32)

noisy_train, noisy_test, clean_train, clean_test = train_test_split(noisy_processed_32, clean_processed_32, test_size=0.2, random_state=42)
params_train, params_test = train_test_split(res, test_size=0.2, random_state=42)

#%% 

plt.imshow(pixel_map_32[1].T, cmap = "hot", origin = "lower", norm = LogNorm(vmin = 0.01, vmax = 10000))
plt.colorbar()

#%% Spectrum reconstruction Code

q = constants.e
y = 10 * 35/32 * 1e-3
B = 100 * 1e-3 
L = 50 * 1e-3
D = 100 * 1e-3 
m = 938.28 * 1e6 * constants.e / constants.c ** 2

Num_data_32 = 15
Num_data_64 = 33
image_dimensions = [0, 35 * 1e-3]

"""
Code to plot out the fitted spectrum
"""
# = pixel_map_64[9]
#mf.plot_spectrum(test_image, image_dimensions, q, B, L, D, m, Num_data)

#    E_kin = (q * B * L * (D + L/2)) ** 2 / (2 * m * y ** 2)
 #   E_kin_MeV = E_kin / (1e6 * constants.e)
Predictions_32 = mf.Image_analysis(pixel_map_32, image_dimensions, q, B, L, D, m, Num_data_32) 
Predictions_64 = mf.Image_analysis(pixel_map_64, image_dimensions, q, B, L, D, m, Num_data_64) 

Errors_T_32 = ml.Evaluate_model(np.array(Predictions_32)[:,0], np.array(res)[:,0])
Errors_N_32 = ml.Evaluate_model(np.array(Predictions_32)[:,1], np.array(res)[:,1])
Errors_E_max_32 = ml.Evaluate_model(np.array(Predictions_32)[:,2], np.array(res)[:,2])

Errors_T_64 = ml.Evaluate_model(np.array(Predictions_64)[:,0], np.array(res)[:,0])
Errors_N_64 = ml.Evaluate_model(np.array(Predictions_64)[:,1], np.array(res)[:,1])
Errors_E_max_64 = ml.Evaluate_model(np.array(Predictions_64)[:,2], np.array(res)[:,2])

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(9, 4))
plt.subplots_adjust(wspace=0, hspace=0)

ax1.set_ylabel("Percentage Error / %")
bplot1 = ax1.boxplot([Errors_T_32["% Errors"], Errors_T_64["% Errors"]], patch_artist=True, labels = ["32 x 32", "64 x 64"])
bplot2 = ax2.boxplot([Errors_N_32["% Errors"], Errors_N_64["% Errors"]], patch_artist=True, labels = ["32 x 32", "64 x 64"])
bplot3 = ax3.boxplot([Errors_E_max_32["% Errors"], Errors_E_max_64["% Errors"]], patch_artist=True, labels = ["32 x 32", "64 x 64"])

ax2.get_yaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)

colors = ['lightblue', 'pink']
for bplot in (bplot1, bplot2, bplot3):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
ax1.set_ylim(0, 50)
ax2.set_ylim(0, 50)
ax3.set_ylim(0, 50)
ax1.set_title("Temperature")
ax2.set_title("No of Particles")
ax3.set_title("Maximum energy")