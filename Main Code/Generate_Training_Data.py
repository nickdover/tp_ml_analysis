# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 03:00:05 2022

@author: user
"""
#%%
from Data_Synthesis import ThomsonParabolaSpec, OutputPlane, Beam, Run, Image
import random
import pickle
import os
import time

def GenerateDataset(Beam, Thomson, Detector, thermal_E_MeV, Num_particles, E_max_MeV, dataset_size, foldername, x_range_mm, y_range_mm, Propagation_step_size = 9 * 10 ** -11):
    config = dict()
    config["Number of particles range"] = Num_particles
    config["Thermal energy range (MeV)"] = thermal_E_MeV
    config["Maximium energy range (MeV)"] = E_max_MeV
    config["Image x Range (mm)"] = x_range_mm
    config["Image y Range (mm)"] = y_range_mm
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    with open(f"{foldername}/Simulation Configurations.obj","wb") as f0:
        pickle.dump(config, f0)
        
    for i in range(dataset_size):
        dataset = dict()
        start = time.time()
        
        E_MeV = random.uniform(thermal_E_MeV[0], thermal_E_MeV[1])
        E_max = random.uniform(E_max_MeV[0], E_max_MeV[1])
        Num = random.randint(Num_particles[0], Num_particles[1])
        
        dataset["Thermal energy of beam (MeV)"] = E_MeV
        dataset["Maximum energy of beam (MeV)"] = E_max
        dataset["Number of particles in beam"] = Num 
        
        Particles = Beam.generate_beam(Num, E_MeV, E_max)
    
        res = Run(Particles, Thomson, Detector, Propagation_step_size)
        
        data = res.copy()
        dataset["Number of accepted particles in beam"] = len(data[0])

            
        h = Image.generate_image(data[0], data[1], data[2], [32, 32], x_range_mm, y_range_mm)
        j = Image.generate_image(data[0], data[1], data[2], [64, 64], x_range_mm, y_range_mm)
        k = Image.generate_image(data[0], data[1], data[2], [128, 128], x_range_mm, y_range_mm)
        dataset["Pixels map 32 x 32"] = h.copy()
        dataset["Pixels map 64 x 64"] = j.copy()
        dataset["Pixels map 128 x 128"] = k.copy()
        print(f"Image {i} completed")
        end = time.time()
        dataset["Run time(s)"] = end - start

        with open(f"{foldername}/{time.time()}.obj","wb") as f0:
            pickle.dump(dataset, f0)


 #%% System Initialisation 

#Define System Here

Collimator_mm = 120
Collimator_diameter_micron = 200

E_strength_kV = 2
B_strength_mT = r'circular_magnet.mat'
Thomson_Plane_mm = 170
dimensions_mm = [10, 50, 50]

Detector_Plane_mm = 320
Detector_dimensions_mm = [100, 100]

#Define Beam Here
mass_MeV = 938.28
charge_e = 1

# Define Image parameters here
x_range_mm = [0, 35]
y_range_mm = [0, 35]

# learning parameters    (Need to input a range)
thermal_E_MeV = [0.1, 6]    #Teff
Num_particles = [4000, 8000]
E_max_MeV = [4, 10]

dataset_size = 10000
filename = "Dataset 2"

Thomson = ThomsonParabolaSpec(E_strength_kV, B_strength_mT, Thomson_Plane_mm, dimensions_mm)
Detector = OutputPlane(Detector_Plane_mm, Detector_dimensions_mm)
Proton_beam = Beam(Collimator_mm, Collimator_diameter_micron, mass_MeV, charge_e)

dataset = GenerateDataset(Proton_beam, Thomson, Detector, thermal_E_MeV, Num_particles, E_max_MeV, dataset_size, filename, x_range_mm, y_range_mm)

