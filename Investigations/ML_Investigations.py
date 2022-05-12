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
import Median_Filter as mf
import pickle

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
x_range_mm = [0, 35]
y_range_mm = [0, 35]

weight = 0.1
background_noise = 10
abberations = 0.2
hard_hit_noise_fraction = 0.1
detector_threshold = 255
zero_point_micron = 200
multiplier = 1

noisy_data_64, res = ml.add_noise(pixel_map_64, res, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
clean_data_64, _ = ml.add_noise(pixel_map_64, res, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
# Initial Processing of data 
noisy_processed_64 = ml.preprocess(noisy_data_64)
clean_processed_64 = ml.preprocess(clean_data_64)

noisy_train, noisy_test, clean_train, clean_test = train_test_split(noisy_processed_64, clean_processed_64, test_size=0.2, random_state=42)
params_train, params_test = train_test_split(res, test_size=0.2, random_state=42)

#%%  Validation with clean data 
model, history = ml.getModel("64x64 Noisy model")


#%%
q = constants.e
y = 10 * 35/32 * 1e-3
B = 100 * 1e-3 
L = 50 * 1e-3
D = 100 * 1e-3 
m = 938.28 * 1e6 * constants.e / constants.c ** 2
image_dimensions = [0, 35 * 1e-3]
Num_data_32 = 6
Discard_data_32 =12
plt.rcParams.update({'font.size': 14}) 

#Predictions = scaler.inverse_transform(Clean_32_model.predict(clean_test))
Predictions_tra = mf.Image_analysis(clean_test * 2550, image_dimensions, q, B, L, D, m, Num_data_32, Discard_data_32) 

_, clean_map = train_test_split(pixel_map_32, test_size=0.2, random_state=42)
counts = []
for image in clean_map:
    counts.append(int(sum(sum(image))))

Errors_T_32 = ml.Evaluate_model(np.array(Predictions)[:,0], np.array(params_test)[:,0])
Errors_N_32 = ml.Evaluate_model(np.array(Predictions)[:,1], np.array(params_test)[:,1])
Errors_E_max_32 = ml.Evaluate_model(np.array(Predictions)[:,2], np.array(params_test)[:,2])


Errors_T_32_tra = ml.Evaluate_model(np.array(Predictions_tra)[:,0], np.array(params_test)[:,0])
Errors_N_32_tra = ml.Evaluate_model(np.array(Predictions_tra)[:,1], np.array(counts))
Errors_E_max_32_tra = ml.Evaluate_model(np.array(Predictions_tra)[:,2], np.array(params_test)[:,2])

#%%
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))
plt.rcParams.update({'font.size': 10}) 

bplot1 = ax1.boxplot([Errors_T_32_tra["% Errors"], Errors_T_32["% Errors"]], patch_artist=True, labels = ["Traditional","CNN"], showfliers=False)
bplot2 = ax2.boxplot([Errors_N_32_tra["% Errors"], Errors_N_32["% Errors"]], patch_artist=True, labels = ["Traditional", "CNN"], showfliers=False)
bplot3 = ax3.boxplot([Errors_E_max_32_tra["% Errors"], Errors_E_max_32["% Errors"]], patch_artist=True, labels = ["Traditional", "CNN"], showfliers=False)
ax1.set_ylabel("Percentage error / % ")


ax1.set_title("Temperature")
ax2.set_title("No of Particles")
ax3.set_title("Maximum energy")

colors = ['lightblue', 'pink']
for bplot in (bplot1, bplot2, bplot3):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)


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

#%%
q = constants.e
y = 10 * 35/32 * 1e-3
B = 100 * 1e-3 
L = 50 * 1e-3
D = 100 * 1e-3 
m = 938.28 * 1e6 * constants.e / constants.c ** 2

Num_data_32 = 8
Discard_data_32 = 10
image_dimensions = [0, 35 * 1e-3]
mf.plot_spectrum(pixel_map_32[0], image_dimensions, q, B, L, D, m, Num_data_32, Discard_data_32)

#%%    
with open("64_128_64CNN_64Pixels_NOISY.obj","rb") as f0:
    m = pickle.load(f0)
    
plt.rcParams.update({'font.size': 14}) 
Predictions_m = m["Predicted_variables"]
True_m = m["True_variables"]
Errors_T_64_m = ml.Evaluate_model(np.array(Predictions_m)[:,0], np.array(True_m)[:,0])

fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
plt.subplots_adjust(wspace=0, hspace=0)

ax1.set_ylabel("Percentage Error / %")
bplot1 = ax1.boxplot([Errors_T_64_m["% Errors"]], patch_artist=True, labels = ["Multi-tracks"], showfliers=False)

#%%

print(np.percentile(Errors_T_64_m["% Errors"], 75) - np.percentile(Errors_T_64_m["% Errors"], 25))
#%%
import time

Predictions = scaler.inverse_transform(Optimised_model.predict(noisy_test))



#%%
start = time.time()
filtered_imge = []
for image in noisy_test:
    filtered_imge.append(mf.remove_noise(image.squeeze()))
end = time.time()
print((start - end) / len(Predictions))
#%%
q = constants.e
y = 10 * 35/32 * 1e-3
B = 100 * 1e-3 
L = 50 * 1e-3
D = 100 * 1e-3 
m = 938.28 * 1e6 * constants.e / constants.c ** 2

Num_data = 10
Discard_data = 0
#%%
Predictions_tra = mf.Image_analysis(np.array(filtered_imge[0:60]) * 2550, image_dimensions, q, B, L, D, m, Num_data, Discard_data)
#%%
plt.rcParams.update({'font.size': 11}) 

Errors_T_64 = ml.Evaluate_model(np.array(Predictions)[:,0], np.array(params_test)[:,0])
Errors_N_64 = ml.Evaluate_model(np.array(Predictions)[:,1], np.array(params_test)[:,1])
Errors_E_max_64 = ml.Evaluate_model(np.array(Predictions)[:,2], np.array(params_test)[:,2])
#%%
Errors_T_64_t = ml.Evaluate_model(np.array(Predictions_tra)[:,0], np.array(params_test[0:60])[:,0])
Errors_N_64_t = ml.Evaluate_model(np.array(Predictions_tra)[:,1], np.array(params_test[0:60])[:,1])
Errors_E_max_64_t = ml.Evaluate_model(np.array(Predictions_tra)[:,2], np.array(params_test[0:60])[:,2])

Error_E_max_traditional = [x for x in Errors_E_max_64_t["% Errors"] if str(x) != 'nan']



fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(13, 4))

bplot1 = ax1.boxplot([Errors_T_64_t["% Errors"], Errors_T_64["% Errors"]], patch_artist=True, labels = ["Traditional", "CNN"], showfliers=False)
bplot2 = ax2.boxplot([Errors_N_64_t["% Errors"], Errors_N_64["% Errors"]], patch_artist=True, labels = ["Traditional", "CNN"], showfliers=False)
bplot3 = ax3.boxplot([Error_E_max_traditional, Errors_E_max_64["% Errors"]], patch_artist=True, labels = ["Traditional", "CNN"], showfliers=False)
ax1.set_title("Temperature")
ax2.set_title("No of Particles")
ax3.set_title("Maximum energy")
ax1.set_ylabel("Percentage Error / %")
colors = ['lightblue', 'pink']
for bplot in (bplot1, bplot2, bplot3):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
#%%
print(np.percentile(Errors_E_max_64["% Errors"], 50))
print(np.percentile(Errors_E_max_64["% Errors"], 25) - np.percentile(Errors_E_max_64["% Errors"], 75))
print(np.average(Errors_E_max_64["% Errors"]))
#%%

#Errors = Errors_E_max_64["% Errors"]
Energy_actu = params_test[:,0]
Energy_pred = Predictions[:,0]

Errors = abs(Energy_actu - Energy_pred)
err_0 = function(0, 1, Energy_actu, Errors)        
err_1 = function(1, 2, Energy_actu, Errors)    
err_2 = function(2, 3, Energy_actu, Errors)  
err_3 = function(3, 4, Energy_actu, Errors)   
err_4 = function(4, 5, Energy_actu, Errors) 
err_5 = function(5, 6, Energy_actu, Errors)  
fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
bplot1 = ax1.boxplot([err_0, err_1,err_2, err_3, err_4, err_5], patch_artist=True, labels = ["0-1 MeV", "1-2 MeV", "2-3 MeV", "3-4 MeV", "4-5 MeV","5-6 MeV"], showfliers=False)
ax1.set_xlabel("Predicting temperature range")
ax1.set_ylabel("Absolute error / MeV")
#%%%
def function(lower, upper, E, Err):
    Error_list = []
    for i in range(len(E)):
        if E[i]>= lower and E[i]<= upper:
            Error_list.append(Err[i])
    return Error_list

#%%
pl.rcParams["figure.figsize"] = (6,6)
pl.rcParams.update({'font.size': 14})
Num_data = 10
Discard_data = 10
mf.plot_spectrum(pixel_map_64[2],image_dimensions, q, B, L, D, m, Num_data, Discard_data)




