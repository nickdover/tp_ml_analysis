# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 14:23:03 2022

@author: user
"""
#%%
from scipy.ndimage import median_filter
import statistics as stat
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import structural_similarity as ssim
import scipy.constants as constants
from scipy.optimize import curve_fit
#%%

def remove_noise(noisy_map):
    def median():
        f = median_filter(noisy_map, size=(3,3))
        return f
    
    def remove_background(f):
        most_common = stat.mode(f.flatten())
        f -= most_common
        return f # subtract most common element

    filtered_map = median()
    filtered_map = remove_background(filtered_map)
    
    return filtered_map

def plot_maps(noisy_map, filtered_map, noisefree_map, title="Median filtered 3x3"):
    fig, axs = plt.subplots(1,3)
    axs[0].imshow(noisy_map.T, cmap='hot', origin='lower')
    axs[0].set_title('Noisy')
    axs[1].imshow(filtered_map.T, cmap='hot', origin='lower')
    axs[1].set_title('Filtered')
    axs[2].imshow(noisefree_map.T, cmap='hot', origin='lower') # plot the thresholded version
    axs[2].set_title('Noise-free')
    fig.suptitle(title)
    
def evaluate(filtered_map, noisefree_map, threshold, track_only=False):
    def mean_squared_error():
        total = 0
        pixels = 0 # total number of pixels in the image
        map_size = filtered_map.shape[0]
        for i in range(map_size):
            for j in range(map_size):
                diff = filtered_map[i][j] - noisefree_map[i][j]
                
                if track_only:
                    #  unless  both are zero, otherwise count
                    if (filtered_map[i][j] != 0 and noisefree_map[i][j] == 0) or (noisefree_map[i][j] != 0 and filtered_map[i][j] == 0): 
                        total += diff ** 2
                        pixels += 1
                else:
                    total += diff ** 2
                    pixels += 1
                    
        error = total / pixels
        
        return error
                
    def signal_noise():
        mse = mean_squared_error()
        ratio = 10 * np.log10(threshold ** 2 / mse)
        
        return ratio
    
    signal_to_noise_ratio = signal_noise()
    SSIM = ssim(noisefree_map, filtered_map, data_range=filtered_map.max() - filtered_map.min()) # otherwise treat as -1 to 1 
    
    print(f"Signal-to-noise ratio (SNR): {signal_to_noise_ratio}")
    print(f"Structural similarity index measure (SSIM): {SSIM}")
    
    # function is checked by comparing two noise free images
    # SSIM = 1 as expected
    
# warning: this function uses the parabolic equation - assumed uniform E and B field
def reconstruct_spectrum(pixel_map, image_dimensions, q, B, L, D, m, Num_data):
    Num_pixels = len(pixel_map)
    pixel_edges = np.linspace(image_dimensions[0], image_dimensions[1], Num_pixels + 1)
    
    bin_edges = []
    for pixel_edge in pixel_edges:
        if pixel_edge != 0:
            E_kin = (q * B * L * (D + L/2)) ** 2 / (2 * m * pixel_edge ** 2)
            E_kin_MeV = E_kin / (1e6 * constants.e)
        else: 
            E_kin_MeV = 10000   # arbitarialy large energy for the 0th pixel
        bin_edges.append(E_kin_MeV)
    bin_height = []
    bin_centres = []
    width = []
    for i in range(Num_data):
        count = sum(pixel_map[:,i])
        bin_height.append(count / (bin_edges[i] - bin_edges[i + 1]))
        bin_centres.append((bin_edges[i] + bin_edges[i + 1])/2)
        width.append((bin_edges[i] - bin_edges[i + 1])/2)
        
    count = 0
    for j in range(Num_data, Num_pixels):
        count += sum(pixel_map[:,j])
    bin_centres.append((bin_edges[Num_data] + bin_edges[Num_pixels])/2)
    bin_height.append(count / (bin_edges[Num_data] - bin_edges[Num_pixels]))
    width.append((bin_edges[Num_data] - bin_edges[Num_pixels])/2)
    return bin_centres[1:], bin_height[1:], width[1:]

def Boltzman(x, N, E):
    return N * np.exp(-x / (E))

def Image_analysis(Images, image_dimensions, q, B, L, D, m, Num_data):
    res = []
    for Image in Images:
        bin_centres, bin_height, width = reconstruct_spectrum(Image, image_dimensions, q, B, L, D, m, Num_data)
        popt, pcov = curve_fit(Boltzman, bin_centres, bin_height, p0 = [2000, 3])
        Num_particles = sum(sum(Image))
        E_MeV = popt[1]
        A = popt[0]
        E_max = -E_MeV * np.log(1 - Num_particles / (A * E_MeV))
        res.append([E_MeV, Num_particles, E_max])
            
    return res

def plot_spectrum(image, image_dimensions, q, B, L, D, m, Num_data):
    bin_centres, bin_height, width = reconstruct_spectrum(image, image_dimensions, q, B, L, D, m, Num_data)
    plt.errorbar(bin_centres, bin_height, xerr = width, fmt =  "x", color = "black")
    popt, pcov = curve_fit(Boltzman, bin_centres, bin_height, p0 = [2000, 3])
    x = np.arange(0, bin_centres[0] + width[0], 0.01)
    plt.plot(x, Boltzman(x, *popt))
    #plt.xscale("log")
    plt.xlabel("Energy / MeV")
    plt.ylabel("Number of particles")
    plt.show()