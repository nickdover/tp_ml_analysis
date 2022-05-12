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
from keras_tuner import Hyperband
#%% Data retreival & Data Processing
folders = ["D:\Year 3 Project\Year-3-Project\Dataset_2"]
#ml.unzip(foldername)
pixel_map_32, pixel_map_64, res = [], [], []
for folder in folders:
    dataset = ml.getData(folder, ["Thermal energy of beam (MeV)", 'Number of accepted particles in beam', "Maximum energy of beam (MeV)"])
    pixel_map_32 += dataset[0]
    pixel_map_64 += dataset[1]
    res += dataset[2]

x_range_mm = [0, 35]
y_range_mm = [0, 35]

weight = 0.1 
background_noise = 10
abberations = 0.2
hard_hit_noise_fraction = 0.1
detector_threshold = 255
zero_point_micron = 200
multiplier = 1

noisy_data, res = ml.add_noise(pixel_map_64, res, multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
clean_data, _ = ml.add_noise(pixel_map_64, res, multiplier, weight, 0, 0, 0, detector_threshold, 0, x_range_mm, y_range_mm)
# Initial Processing of data 
noisy_processed = ml.preprocess(noisy_data)
clean_processed = ml.preprocess(clean_data)

noisy_train, noisy_test, clean_train, clean_test = train_test_split(noisy_processed, clean_processed, test_size=0.2, random_state=42)
params_train, params_test = train_test_split(res, test_size=0.2, random_state=42)

#%% Model Training

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


#%%
epochs = 50
learning_rate = 1e-4

model_name = "CNN 64 x 64"

"""
Note: The model is trained on parameters with MinMaxScaler. So need to inverse tranform 
the prediction of the model. 

i.e scaler.inverse_transform(model.predict(noisy_test))

"""

model, history, scaler = ml.Train_model(clean_train, params_train, Optimised_model, 
                                        epochs = epochs, learning_rate = learning_rate, 
                                        store_model = True, model_name = model_name)


#%% Hyperparameter fine-tuning

def build_model(hp):
    """
    Tune hyperparameters to minimise loss value of model
    
    Code reference: 
        
    https://keras.io/keras_tuner/#:~:text=KerasTuner%20is%20an%20easy%2Dto,hyperparameter%20values%20for%20your%20models.

    """
    model = keras.Sequential(
        [
            keras.Input(shape=((64,64, 1))),
            layers.Conv2D(filters = hp.Choice(
                'num_filters',
                values=[8, 16, 32, 64, 128],
                default=16), kernel_size=(3, 3), activation="relu", padding="same"),
            
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters = hp.Choice(
                'num_filters_2',
                values=[8, 16, 32, 64, 128],
                default=16), kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(filters = hp.Choice(
                'num_filters_3',
                values=[8, 16, 32, 64, 128],
                default=16), kernel_size=(3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(rate=hp.Float(
                'dropout_1',
                min_value=0.0,
                max_value=0.35,
                default=0.2,
                step=0.05,
            )),
            layers.Flatten(),
            layers.Dense(units=hp.Int("units_1", min_value=0, max_value=128, step=16),
            activation="relu",
        ),
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















