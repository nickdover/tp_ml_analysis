# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:13:42 2022

@author: user
"""
#%%
from tensorflow import keras
from tensorflow.keras import layers
import Machine_Learning as ml
from sklearn.model_selection import train_test_split

#%% Data retreival 
folders = ["Dataset_2", "Dataset_3"]
#ml.unzip(foldername)
pixel_map_32, pixel_map_64, res = [], [], []
for folder in folders:
    dataset = ml.getData(folder, ["Thermal energy of beam (MeV)", 'Number of accepted particles in beam', "Maximum energy of beam (MeV)"])
    pixel_map_32 += dataset[0]
    pixel_map_64 += dataset[1]
    res += dataset[3]


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

noisy_data_32, res = ml.add_noise(pixel_map_32, res, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
clean_data_32, _ = ml.add_noise(pixel_map_32, res, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
# Initial Processing of data
noisy_processed_32 = ml.preprocess(noisy_data_32)
clean_processed_32 = ml.preprocess(clean_data_32)

noisy_train, noisy_test, clean_train, clean_test = train_test_split(noisy_processed_32, clean_processed_32, test_size=0.2, random_state=42)
params_train, params_test = train_test_split(res, test_size=0.2, random_state=42)

#%% Model Training

batch_size = 128
learning_rate = 0.005
epochs = 5

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
print("Test binary_crossentropy (loss):", score)
