# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 21:06:35 2022

@author: user
"""
#%%
from tensorflow import keras
from tensorflow.keras import layers
import Machine_Learning as ml
from sklearn.model_selection import train_test_split
from kerastuner.tuners import Hyperband

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

Optimised_model = keras.Sequential(
    [
        keras.Input(shape=(32,32, 1)),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(16, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(32, activation="relu"),
        layers.Dense(3, activation="linear"),                                                                            
    ]
)  

epochs = 50
learning_rate = 1e-4
model, history, scaler = ml.Train_model(noisy_train, params_train, Optimised_model, epochs = epochs, learning_rate = learning_rate)

#%% Hyperparameter fine-tuning

def build_model(hp):
    """
    Tune hyperparameters to minimise loss value of model
    
    Code reference: 
        
    https://keras.io/keras_tuner/#:~:text=KerasTuner%20is%20an%20easy%2Dto,hyperparameter%20values%20for%20your%20models.

    """
    model = keras.Sequential(
        [
            keras.Input(shape=((32,32, 1))),
            layers.Conv2D(filters = hp.Choice(
                'num_filters',
                values=[8, 16, 32, 64, 128, 256],
                default=16), kernel_size=(3, 3), activation="relu", padding="same"),
            
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters = hp.Choice(
                'num_filters_2',
                values=[8, 16, 32, 64, 128, 256],
                default=16), kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters = hp.Choice(
                'num_filters_3',
                values=[8, 16, 32, 64, 128, 256],
                default=16), kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters = hp.Choice(
                'num_filters_4',
                values=[8, 16, 32, 64, 128, 256],
                default=16), kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            )),
            layers.Flatten(),
            layers.Dense(units=hp.Int("units_1", min_value=0, max_value=128, step=16),
            activation="relu",
        ),
            layers.Dropout(rate=hp.Float(
                'dropout_2',
                min_value=0.0,
                max_value=0.5,
                default=0.25,
                step=0.05,
            )),
            layers.Dense(units=hp.Int("units_2", min_value=0, max_value=128, step=16), activation="relu"),
            layers.Dense(3, activation="linear"),                                                                            
        ]
    )   
    model.compile(optimizer=keras.optimizers.Adam(1e-4),loss='mse', metrics=['mae'])
    return model

x_train, x_val, y_train, y_val = train_test_split(noisy_train, params_train, test_size=0.2, random_state=42)
tuner = Hyperband(build_model,objective='val_loss', project_name="best_noisy_data_model")
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))    
best_model = tuner.get_best_models()[0]

#%% Find Optimal learning rate















