# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 00:58:08 2022

@author: user
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 16:42:49 2022

@author: user
"""
#%% Recover Dataset
import os 
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import glob
import pickle 
from Data_Synthesis import Image
import tarfile
import progressbar
from sklearn.preprocessing import MinMaxScaler  
#%%
def unzip(foldername):
    tar = tarfile.open(f"{foldername}.tar", "r")
    tar.extractall()
    
def getData(foldername, res_variables = ["Thermal energy of beam (MeV)"]):
    pixel_map_32 = []
    pixel_map_64 = []
    res = []
    path = f'{foldername}'
    files = glob.glob(os.path.join(path, '*.obj'))
    print(f"Total number of files {len(files)}")
    widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
    progressbar.Counter(format='%(value)02d/%(max_value)d')
]
    bar = progressbar.ProgressBar(max_value=len(files), widgets= widgets)
    init = 0

    for filename in files:
        with open(filename,"rb") as f0:
            if filename != path + "\Simulation Configurations.obj":
                dataset = pickle.load(f0)   
                try: 
                    pixel_map_32.append(dataset["Pixels map 28 x 28"])  # Defined incorrectly in the datafile; need to change after
                    pixel_map_64.append(dataset["Pixels map 64 x 64"])
                    y = []
                    for var in res_variables:
                        y.append(dataset[var])
                    res.append(y)
                except: 
                    print(f"Corrupt data file {filename}")
            init += 1
            bar.update(init)

    return pixel_map_32, pixel_map_64, res

def add_noise(pixel_map, T, dataset_multiplier, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, Collimator_diameter_micron, x_range, y_range):
    noisy_pixel_map = []
    T_noisy = []
    print(f"Total Dataset Size: {len(pixel_map)}")
    print(f"Mutiplier: {dataset_multiplier}")
    Total_size = len(pixel_map) * dataset_multiplier
    print(f"Noisy Dataset Size: {Total_size}")
    
    widgets=[
    ' [', progressbar.Timer(), '] ',
    progressbar.Bar(),
    ' (', progressbar.ETA(), ') ',
    progressbar.Counter(format='%(value)02d/%(max_value)d')
]
    bar = progressbar.ProgressBar(max_value=Total_size, widgets= widgets)
    init = 0
    for h in range(len(pixel_map)):
        for i in range(dataset_multiplier):
            res = Image.add_noise(pixel_map[h], weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, Collimator_diameter_micron, x_range, y_range)
            noisy_pixel_map.append(res.copy())
            T_noisy.append(T[h])
            init += 1
            bar.update(init)
    return np.array(noisy_pixel_map), np.array(T_noisy)

def Train_model(x_train, y_train, model, epochs, learning_rate = 0.1, loss = "mean_squared_error", metrics = ['mean_absolute_percentage_error', "mean_absolute_error"]
, store_model = False, model_name = ""):
    
    if type(y_train[0]) == np.float64:
        y_train = np.expand_dims(y_train, -1)
    scaler = MinMaxScaler() 
    scaler.fit(y_train)
    y_train = scaler.transform(y_train)
    
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    
    model.summary()
    batch_size = 128
    epochs = epochs

    opt = keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss=loss, optimizer=opt, metrics=metrics)
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    
    if store_model:
        model.save(model_name+".h5")
        with open(f"{model_name}.obj","wb") as f0:
            pickle.dump(history.history, f0)
            
    return model, history, scaler

def getModel(model_name):
    new_model = keras.models.load_model(model_name+".h5")
    with open(f"{model_name}.obj","rb") as f0:
        history = pickle.load(f0)
    return new_model, history

def plot_loss(history):
    Epoch = np.arange(1, len(history['loss']) + 1, 1)
    plt.plot(Epoch, history['loss'], label='loss')
    plt.plot(Epoch, history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [Mean Squared Error]')
    plt.legend()
    plt.grid(True)
    
def Evaluate_model(y_pred, y_actu):
    def absPercentageError(actual, predictions):
        res = []
        for i in range(len(actual)):
            res.append(abs(actual[i] - predictions[i]) / predictions[i] * 100) 
        return res 

    def SqaredError(actual, predictions):
        res = []
        for i in range(len(actual)):
            res.append((actual[i] - predictions[i])**2) 
        return res

    def AbsError(actual, predictions):
        res = []
        for i in range(len(actual)):
            res.append(abs(actual[i] - predictions[i])) 
        return res
    
    dataset = dict()
    dataset["% Errors"] = absPercentageError(y_pred, y_actu)
    dataset["Squared Errors"] = SqaredError(y_pred, y_actu)
    dataset["Abs Errors"] = AbsError(y_pred, y_actu)
    
    return dataset

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """
    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), array.shape[1], array.shape[2], 1))
    return array