#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:44:07 2024

@author: amir.m
"""


import numpy as np
from scipy.stats import entropy

# Function to compute entropy
def compute_entropy(data):
    # Flatten the data and compute the histogram
    data_flat = data.flatten()
    hist, _ = np.histogram(data_flat, bins=100, density=True)
    
    # Remove zeros from the histogram to avoid issues with log calculation
    hist = hist[hist > 0]
    
    # Compute entropy
    ent = entropy(hist)
    return ent

# Load features
stft_features = np.load('tb_logs/stft_features.npy', allow_pickle=True)
logmel_features = np.load('tb_logs/logmel_features.npy', allow_pickle=True)

# Remove extra dimension from Log Mel Features
logmel_features = logmel_features.squeeze(1) 
logmel_features = np.transpose(logmel_features, (0, 2, 1))

# Select one sample from each feature type
stft_sample = stft_features[0]  
logmel_sample = logmel_features[0]  



import matplotlib.pyplot as plt

# Plot STFT
plt.figure(figsize=(8, 6))
plt.imshow(stft_sample, aspect='auto', cmap='gray', origin='lower')
plt.colorbar(label='Magnitude')

# Set title and labels with increased font size
plt.title('STFT Feature', fontsize=16)
plt.xlabel('Time Frames', fontsize=14)
plt.ylabel('Frequency Bins', fontsize=14)

# Adjust tick parameters for x and y axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig('features/stft_feature_sample.png', dpi=300)
plt.close()

# Plot Log Mel Filter Bank
plt.figure(figsize=(8, 6))
plt.imshow(logmel_sample, aspect='auto', cmap='gray', origin='lower')
plt.colorbar(label='Magnitude')

# Set title and labels with increased font size
plt.title('Log Mel Filter Bank Feature', fontsize=16)
plt.xlabel('Time Frames', fontsize=14)
plt.ylabel('Mel Frequency Bins', fontsize=14)

# Adjust tick parameters for x and y axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig('features/logmel_feature_sample.png', dpi=300)
plt.close()




# Compute entropy
stft_entropy = compute_entropy(stft_sample)
logmel_entropy = compute_entropy(logmel_sample)

# Print the results
print(f'STFT Sample Entropy: {stft_entropy:.4f}')
print(f'Log Mel Filter Bank Sample Entropy: {logmel_entropy:.4f}')


def compute_entropy_multiple(data_samples):
    entropies = []
    for sample in data_samples:
        # Flatten the data and compute the histogram
        data_flat = sample.flatten()
        hist, _ = np.histogram(data_flat, bins=100, density=True)
        
        # Remove zeros from the histogram to avoid issues with log calculation
        hist = hist[hist > 0]
        
        # Compute entropy
        ent = entropy(hist)
        entropies.append(ent)
    return np.mean(entropies)

# Load features
stft_features = np.load('stft_features.npy', allow_pickle=True)
logmel_features = np.load('logmel_features.npy', allow_pickle=True)

# Compute entropy for multiple samples
stft_entropy_multiple = compute_entropy_multiple(stft_features)
logmel_entropy_multiple = compute_entropy_multiple(logmel_features)

# Print the results
print(f'Average STFT Entropy: {stft_entropy_multiple:.4f}')
print(f'Average Log Mel Filter Bank Entropy: {logmel_entropy_multiple:.4f}')
