#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 16:59:49 2024

@author: amir.m
"""

import os
import numpy as np
import torch
import scipy.io.wavfile as wav

from nnAudio import features
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

# Paths
base_dir = 'Datasets/DeepShip/Segments_5s_16000hz'
train_paths_file = 'train_paths.txt'

# Parameters
sr = 16000
n_fft = 1024
hop_length_stft = 1024  
hop_length_mel = 160  
n_mels = 64
freq_bins = 48  

# Initialize feature extraction modules
stft_transform = features.STFT(sr=sr, n_fft=n_fft, hop_length=hop_length_stft, fmin=1e-6, fmax=sr // 2,
                               win_length=n_fft, freq_bins=48, output_format='Magnitude',trainable=False)
log_stft_transform = features.STFT(sr=sr, n_fft=n_fft, hop_length=hop_length_stft, fmin=1e-6, fmax=sr // 2,
                                   win_length=n_fft, freq_bins=48, freq_scale='log2', output_format='Magnitude',trainable=False)

spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length_mel,
                                    win_length=n_fft, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)

mel_transform = LogmelFilterBank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr // 2, freeze_parameters=True)




ref = 1.0
amin = 1e-10
top_db = None
s_spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=1000,
                                      win_length=n_fft, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
s_mel_transform = LogmelFilterBank(sr=sr, n_fft=n_fft, n_mels=48, fmin=10, fmax=8000,
                                   ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)



# Function to extract features
def extract_features(file_path):
    sr, y = wav.read(file_path)
    y = y.astype(np.float32)
    y_tensor = torch.tensor(y).unsqueeze(0)  # add batch dimension
    
    stft_feature = stft_transform(y_tensor).numpy()
    log_stft_feature = log_stft_transform(y_tensor).numpy()
    spectrogram = spectrogram_transform(y_tensor)
    s_spectrogram = s_spectrogram_transform(y_tensor)
    logmel_feature = mel_transform(spectrogram).numpy()
    s_logmel_feature = s_mel_transform(s_spectrogram).numpy()
    
    return stft_feature.squeeze(0), log_stft_feature.squeeze(0), logmel_feature.squeeze(0), s_logmel_feature.squeeze(0)

# Read train paths
train_paths = []
count = 0
with open(train_paths_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('Train indices and paths:'):
            # Skip the header line
            continue
        if ':' in line:  # Lines with indices (e.g., "0: path/to/file.wav")
            path = line.split(': ')[1]  # Extract the path after the colon
        else:  # Lines without indices (in case your txt format changes)
            path = line

        # Adjust the sample rate in the path (replace 32000hz with 16000hz)
        path = path.replace('32000hz', '16000hz')

        # Append the path to the list
        train_paths.append(path)
        count += 1
print('\ntrain count: ', count)

features = {'STFT': [], 'log_STFT': [], 'LogMel': [], 's_LogMel': []}

# Collect features from all paths
for file_path in train_paths:
    stft_feature, log_stft_feature, logmel_feature, s_logmel_feature = extract_features(file_path)
    
    # Flatten features to 1D to compute a single mean and std
    features['STFT'].append(stft_feature.flatten())
    features['log_STFT'].append(log_stft_feature.flatten())
    features['LogMel'].append(logmel_feature.flatten())
    features['s_LogMel'].append(s_logmel_feature.flatten())

# Compute one mean and std for each feature across all samples
stat_count = 0
statistics = {}
for feature_name in features:
    # Concatenate along the first axis and flatten to a 1D array
    feature_array = np.concatenate(features[feature_name], axis=0)
    
    # Calculate overall mean and std
    mean = np.mean(feature_array)
    std = np.std(feature_array)
    
    statistics[feature_name] = {'mean': mean, 'std': std}
    stat_count += 1
    print(f'\nfeature stat computed: {stat_count}')

print('\n')
# Save statistics to a file
with open('feature_statistics.txt', 'w') as f:
    for feature_name, stats in statistics.items():
        f.write(f'{feature_name} mean: {stats["mean"]}\n')
        f.write(f'{feature_name} std: {stats["std"]}\n\n')
