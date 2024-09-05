#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 16:42:42 2024

@author: amir.m
"""

import numpy as np
from scipy.stats import entropy
import pdb


# Load features
stft_features = np.load('tb_logs/stft_features.npy', allow_pickle=True)
logmel_features = np.load('tb_logs/logmel_features.npy', allow_pickle=True)

# Remove extra dimension from Log Mel Features
logmel_features = logmel_features.squeeze(1) 
logmel_features = np.transpose(logmel_features, (0, 2, 1))

# Compute the mean and standard deviation for STFT features
stft_mean = np.mean(stft_features)
stft_std = np.std(stft_features)

# Compute the mean and standard deviation for Log Mel features
logmel_mean = np.mean(logmel_features)
logmel_std = np.std(logmel_features)


# Save the results to a text file
with open('tb_logs/features_stats.txt', 'w') as f:
    f.write(f"STFT Features Mean: {stft_mean:.6f}\n")
    f.write(f"STFT Features Standard Deviation: {stft_std:.6f}\n")
    f.write(f"\nLog Mel Filter Bank Features Mean: {logmel_mean:.6f}\n")
    f.write(f"Log Mel Filter Bank Features Standard Deviation: {logmel_std:.6f}\n")

print("Mean and standard deviation have been computed and saved to 'tb_logs/features_stats.txt'.")
