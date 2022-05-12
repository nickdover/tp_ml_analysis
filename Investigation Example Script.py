# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 01:41:22 2022

@author: user
"""
#%%
from Data_Synthesis import ThomsonParabolaSpec, OutputPlane, validatedeflection, Beam, Run, Image, Particle
import numpy as np
import time
import matplotlib.pyplot as pl
import pickle
#%% Parameters for system setup
mass_MeV = 938.28
charge_e = 1
thermal_E_MeV = 3
E_max_MeV = 10
E_min_MeV = 0

Collimator_mm = 120
Collimator_diameter_micron = 200


E_strength_kV = 2
B_strength_mT = 100
#B_strength_mT = r'circular_magnet.mat'
Thomson_Plane_mm = 170
dimensions_mm = [10, 50, 50]

Detector_Plane_mm = 320
Detector_dimensions_mm = [100, 100]

Thomson = ThomsonParabolaSpec(E_strength_kV, B_strength_mT, Thomson_Plane_mm, dimensions_mm)
Detector = OutputPlane(Detector_Plane_mm, Detector_dimensions_mm)

energies = np.arange(0.01, 20, 0.2) 
Proton = []
for energy in energies:
    Proton.append(Particle(mass_MeV, charge_e, energy))
res = Run(Proton, Thomson, Detector, step = 9 * 10 ** -12)

#%%
from scipy import constants
def yE( E_kin, q, B, D, L, m):
    return q * B * L * (D+L/2) / np.sqrt(2 * m * E_kin * 1e6 * constants.e)
#fig, axs = pl.subplots(1, 1,figsize=(14,10))
pl.rcParams.update({'font.size': 14})
pl.rcParams["figure.figsize"] = (6,6)
E = np.linspace(0.1,  20, 100)
analytical = yE(E, constants.e, 100 * 1e-3, 100 * 1e-3 , 50 * 1e-3, 938.28 * 1e6 * constants.e / constants.c ** 2)
pl.plot(np.array(res[1]) * 1e3, res[2], ".", label = "Simulation results", mew = 2, ms =3)
pl.plot(analytical * 1e3, E, label = "Uniform field approximation")
pl.legend()
pl.ylabel("Energy / MeV")
pl.xlabel("Magnetic deflection (y) / mm")
pl.xscale("log")
pl.yscale("log")


#%% Validate model
pl.rcParams["figure.figsize"] = (9,6)
validatedeflection(mass_MeV, charge_e, 0.05, 5, 0.1, Thomson, Detector, "Euler", num_step = 9 * 10 ** -13)

#%% Propagate Proton  

pl.rcParams["figure.figsize"] = (6,6)
pl.rcParams["font.size"] = 14

Propagation_step_size =  10 ** -10
Proton_beam = Beam(Collimator_mm, Collimator_diameter_micron, mass_MeV, charge_e).generate_beam(1000, 4, 10)

x_list, y_list, E_list, x_init_list, y_init_list = Run(Proton_beam, Thomson, Detector, Propagation_step_size)
pl.plot(np.array(x_list) * 1e6, np.array(y_list) * 1e6, "x", label = "Ions")  
pl.legend()
pl.xlabel(r"$x $ / $\mu m$")
pl.ylabel("$y $ / $\mu m$")
pl.show()  
pl.savefig('Figures/4.png', dpi=300)

#%%
def Boltzmann(E, kT):
    return 1 / kT * np.exp(-E/kT)

x = np.arange(0, 10, 0.1)

pl.hist(E_list, bins = 10, density = True, histtype = "step", alpha=0.7, lw = 2.5, label = "Numerical")
pl.plot(x, Boltzmann(x, 4), lw = 2, label = "Boltzmann distribution")
pl.xlabel(r"$E$ / $MeV$")
pl.ylabel("$p(E)$")
pl.legend()
#pl.hist(E_list_2, bins = 10, density = True, histtype = "step")


 #%% Generate Image

x_range_mm = [0, 25]
y_range_mm = [0, 25]
pixels = [28, 28]

weight = 0.6
background_noise = 10
abberations = 0.2
hard_hit_noise_fraction = 0.1
detector_threshold = 255
zero_point_micron = 200

h = Image.generate_image(x_list, y_list, E_list, pixels, x_range_mm, y_range_mm)
#%%
y = Image.add_noise(h, weight, background_noise, abberations, hard_hit_noise_fraction, detector_threshold, zero_point_micron, x_range_mm, y_range_mm)
pl.imshow(y.T, cmap='hot', origin='lower')
pl.colorbar()

#%% Investigate Suitable step size
step_sizes = np.logspace(-13, -9, 20)
error_list = []
computation_time = []
for step_size in step_sizes:
    print(f"Running step size {step_size}")
    start = time.time()
    x_diff, y_diff = validatedeflection(mass_MeV, charge_e, 1, 5, 0.1, Thomson, Detector, "Euler", num_step = step_size)
    end = time.time()
    computation_time.append(end - start)
    error_list.append(x_diff + y_diff)
#%%
pl.plot(error_list, np.array(computation_time) / 40)  
pl.plot(error_list, np.array(computation_time) / 40, "x", label = "Time steps")      
pl.ylabel("Computational time / s")
pl.xlabel("Average absolute error / mm")
pl.xscale("log")
pl.yscale("log")

#%% 
x = np.arange( 1, 5, 0.1)