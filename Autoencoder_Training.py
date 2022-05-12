# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:13:42 2022

@author: user
"""
#%%
"""
This script is for the training of the autoencoder.

Requires a dataset of images stored in a folder. 
Trained model can be stored.    
"""
from tensorflow import keras
from tensorflow.keras import layers
import Machine_Learning as ml
from sklearn.model_selection import train_test_split

#%% Data retreival 
folders = ["D:\Year 3 Project\Year-3-Project\Dataset_2"]
#ml.unzip(foldername)
pixel_map_32, pixel_map_64, res = [], [], []
for folder in folders:
    dataset = ml.getData(folder, ["Thermal energy of beam (MeV)", 'Number of accepted particles in beam', "Maximum energy of beam (MeV)"])
    pixel_map_32 += dataset[0]
    pixel_map_64 += dataset[1]
    res += dataset[2]


#%% Data Processing
x_range_mm = [0, 35]
y_range_mm = [0, 35]

weight = 0.1
background_noise = 10
abberations = 0.2
hard_hit_noise_fraction = 0.1
detector_threshold = 255
zero_point_micron = 200
multiplier = 1

noisy_data, res = ml.add_noise(pixel_map_32, res, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
clean_data, _ = ml.add_noise(pixel_map_32, res, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
# Initial Processing of data
noisy_processed = ml.preprocess(noisy_data)
clean_processed = ml.preprocess(clean_data)

noisy_train, noisy_test, clean_train, clean_test = train_test_split(noisy_processed, clean_processed, test_size=0.2, random_state=42)
params_train, params_test = train_test_split(res, test_size=0.2, random_state=42)

#%% Model Training

batch_size = 128
learning_rate = 0.005
epochs = 20 

autoencoder = keras.Sequential(
    [
        layers.Input(shape=(noisy_train[0].shape)),
        # encoder
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # decoder
        layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, activation="relu", padding="same"),
        layers.Conv2DTranspose(32, kernel_size=(3, 3), strides=2, activation="relu", padding="same"),
        layers.Conv2D(1, kernel_size=(3, 3), activation="sigmoid", padding="same"),
    ]
)
opt = keras.optimizers.Adam(learning_rate = learning_rate)
# Autoencoder
autoencoder.compile(optimizer = opt, loss = "binary_crossentropy")
autoencoder.summary()
history = autoencoder.fit(noisy_train, clean_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, shuffle=True)
score = autoencoder.evaluate(noisy_test, clean_test, verbose=0)
ml.plot_loss(history)
# Model storing so that it can be reused without training model again
model_name = "Autoencoder 32 x 32 "
autoencoder.save(model_name+".h5")
print("Test binary_crossentropy (loss):", score)
#%%
autoencoder, history = ml.getModel("Autoencoder 32 x 32.h5")

