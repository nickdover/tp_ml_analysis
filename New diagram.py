# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:37:51 2022

@author: user
"""
#%%
import Machine_Learning as ml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.constants import constants
import Median_Filter as mf
import pickle
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler  

#%%

folders = ["D:\Year 3 Project\Year-3-Project\Dataset_2"]
#ml.unzip(foldername)
pixel_map_32, pixel_map_64, res = [], [], []
for folder in folders:
    dataset = ml.getData(folder, ["Thermal energy of beam (MeV)", 'Number of accepted particles in beam', "Maximum energy of beam (MeV)"])
    pixel_map_32 += dataset[0]
    pixel_map_64 += dataset[1]
    res += dataset[2]

#%% Noise addition

Analysis_dataset = pixel_map_64

x_range_mm = [0, 35]
y_range_mm = [0, 35]
weight = 0.1 
background_noise = 10
abberations = 0.2
hard_hit_noise_fraction = 0.1
detector_threshold = 255
zero_point_micron = 200
multiplier = 1

noisy_data, res = ml.add_noise(Analysis_dataset, res, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
clean_data, _ = ml.add_noise(Analysis_dataset, res, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
# Initial Processing of data 
noisy_processed = ml.preprocess(noisy_data)
clean_processed = ml.preprocess(clean_data)

noisy_train, noisy_test, clean_train, clean_test = train_test_split(noisy_processed, clean_processed, test_size=0.2, random_state=42)
params_train, params_test = train_test_split(res, test_size=0.2, random_state=42)

#%% Load model 

Optimised_model = keras.Sequential(
    [
        keras.Input(shape=(64,64, 1)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(64, activation="linear"),   
        layers.Dense(3, activation="linear"),                                                                            
    ]
)  

Optimised_model.load_weights("64x64 Noisy model.h5")
scaler = MinMaxScaler() 
scaler.fit(params_train)

#%% Model predict 
CNN_pred = Optimised_model.predict(noisy_test)
CNN_pred = scaler.inverse_transform(CNN_pred)

#%%
from matplotlib.colors import LogNorm
from scipy.signal import savgol_filter

plt.rcParams.update({'font.size': 18}) 
fig, axs = plt.subplots(1, 3, figsize = (20,6))
Error_CNN_p = np.array(abs(CNN_pred - params_test)/params_test * 100)
plt.subplots_adjust(left=-0.03,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, )

axs[0].hist2d(params_test[:,0], Error_CNN_p[:,0], 100, cmap=plt.cm.jet, norm= LogNorm(1, 100))
axs[0].set_ylabel("Absolute error / MeV")
axs[0].set_xlabel("Temperature / MeV")
axs[1].hist2d(params_test[:,1], Error_CNN_p[:,1], 100, cmap=plt.cm.jet, norm= LogNorm(1, 100))
axs[1].set_xlabel("No. Particles")
axs[1].set_ylabel("Absolute error")
axs[1].ticklabel_format(style='sci',scilimits=(0,3),axis='y')
axs[1].set_xlim(4000, 8000)
im = axs[2].hist2d(params_test[:,2], Error_CNN_p[:,2], 100, cmap=plt.cm.jet, norm= LogNorm(1, 100))
axs[2].set_ylabel("Absolute error / MeV")
axs[2].set_xlabel("Maximum energy / MeV")
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.82, 0.1, 0.013, 0.8])
fig.colorbar(im[3], cax=cbar_ax)

plt.show()


#%%

from scipy.stats import skew

Error_tra_p = np.array(abs(Predictions_tra - params_test)/params_test * 100)
cleanedList = [x for x in Error_tra_p[:,2] if str(x) != 'nan']
print(np.average(Error_tra_p[:,0]))
print(np.percentile(Error_tra_p[:,0], 50))
print(np.percentile(Error_tra_p[:,0], 25), np.percentile(Error_tra_p[:,0],75))
print(skew(Error_tra_p[:,0]))

#%%

filtered_image = []
for image in noisy_test:
    filtered_image.append(mf.remove_noise(image.squeeze()))
#%%
N_test = [float(sum(sum(x)))* 2550 for x in clean_test]
Error_N = abs(N_test - params_test[:,1])/params_test[:,1] * 100
#%%
q = constants.e
y = 10 * 35/32 * 1e-3
B = 100 * 1e-3 
L = 50 * 1e-3
D = 100 * 1e-3 
m = 938.28 * 1e6 * constants.e / constants.c ** 2
image_dimensions = [0, 35 * 1e-3]

Num_data = 25
Discard_data = 15
#%%
Predictions_tra = mf.Image_analysis(np.array(filtered_image) * 2550, image_dimensions, q, B, L, D, m, Num_data, Discard_data)
Predictions_tra = np.array(Predictions_tra)
#%%
mf.plot_spectrum(filtered_image[62], image_dimensions, q, B, L, D, m, Num_data, Discard_data)

#%%

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
plt.rcParams.update({'font.size': 10})
bplot1 = ax1.boxplot([Error_tra_p[:,0], Error_CNN_p[:,0]], patch_artist=True, labels = ["Traditional", "CNN"], showfliers=False)
bplot2 = ax2.boxplot([Error_N, Error_CNN_p[:,1]], patch_artist=True, labels = ["Traditional", "CNN"], showfliers=False)
bplot3 = ax3.boxplot([cleanedList, Error_CNN_p[:,2]], patch_artist=True, labels = ["Traditional", "CNN"], showfliers=False)
ax1.set_title("Temperature")
ax2.set_title("No of Particles")
ax3.set_title("Maximum energy")
ax1.set_ylabel("Percentage Error / %")
colors = ['lightblue', 'pink']
for bplot in (bplot1, bplot2, bplot3):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
#%%

with open("64_128_64CNN_64Pixels_NOISY.obj","rb") as f0:
    m = pickle.load(f0)
    
Predictions_m = m["Predicted_variables"]
True_m = m["True_variables"]
Error_m_p = np.array(abs(Predictions_m - True_m)/True_m * 100)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
plt.rcParams.update({'font.size': 10})
bplot1 = ax1.boxplot([Error_CNN_p[:,0], Error_m_p[:,0]], patch_artist=True, labels = ["Single", "Multi"], showfliers=False)
bplot2 = ax2.boxplot([Error_CNN_p[:,1], Error_m_p[:,1]], patch_artist=True, labels = ["Single", "Multi"], showfliers=False)
bplot3 = ax3.boxplot([Error_CNN_p[:,2], Error_m_p[:,2]], patch_artist=True, labels = ["Single", "Multi"], showfliers=False)
ax1.set_title("Temperature")
ax2.set_title("No of Particles")
ax3.set_title("Maximum energy")
ax1.set_ylabel("Percentage Error / %")
colors = ['lightblue', 'pink']
for bplot in (bplot1, bplot2, bplot3):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        
print(np.percentile(Error_m_p[:,2], 50))

