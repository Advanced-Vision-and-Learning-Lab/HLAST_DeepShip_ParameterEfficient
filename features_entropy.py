#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 16:44:07 2024

@author: amir.m
"""

import numpy as np
from scipy.stats import entropy
import pdb

# Load each feature set from its .npz file
stft_data = np.load('tb_logs/stft_features.npz')
stft_features = stft_data['stft_features']

log_stft_data = np.load('tb_logs/log_stft_features.npz')
logmel_data = np.load('tb_logs/logmel_features.npz')
s_logmel_data = np.load('tb_logs/s_logmel_features.npz')

# Access the arrays
log_stft_features = log_stft_data['log_stft_features']
logmel_features = logmel_data['logmel_features']
s_logmel_features = s_logmel_data['s_logmel_features']


# Remove extra dimension from Log Mel Features
logmel_features = logmel_features.squeeze(1) 
logmel_features = np.transpose(logmel_features, (0, 2, 1))

s_logmel_features = s_logmel_features.squeeze(1) 
s_logmel_features = np.transpose(s_logmel_features, (0, 2, 1))


# Select one sample from each feature type
stft_sample = stft_features[0]  
logmel_sample = logmel_features[0]  
log_stft_sample = log_stft_features[0]  
s_logmel_sample = s_logmel_features[0]  

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

# Plot STFT
plt.figure(figsize=(8, 6))
plt.imshow(log_stft_sample, aspect='auto', cmap='gray', origin='lower')
plt.colorbar(label='Magnitude')
# Set title and labels with increased font size
plt.title('log STFT Feature', fontsize=16)
plt.xlabel('Time Frames', fontsize=14)
plt.ylabel('Frequency Bins', fontsize=14)
# Adjust tick parameters for x and y axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('features/log_stft_feature_sample.png', dpi=300)
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

# Plot Log Mel Filter Bank
plt.figure(figsize=(8, 6))
plt.imshow(s_logmel_sample, aspect='auto', cmap='gray', origin='lower')
plt.colorbar(label='Magnitude')
# Set title and labels with increased font size
plt.title('s Log Mel Filter Bank Feature', fontsize=16)
plt.xlabel('Time Frames', fontsize=14)
plt.ylabel('Mel Frequency Bins', fontsize=14)
# Adjust tick parameters for x and y axes
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('features/s_logmel_feature_sample.png', dpi=300)
plt.close()






def compute_histogram(data, bins=100):
    data_flat = data.flatten()
    hist, bin_edges = np.histogram(data_flat, bins=bins, density=True)
    return hist, bin_edges

# Function to normalize data using Min-Max Normalization
def min_max_normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def z_score_normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized_data = (data - mean) / std
    return normalized_data

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



# Compute entropy
stft_entropy = compute_entropy(stft_sample)
logmel_entropy = compute_entropy(logmel_sample)
log_stft_entropy = compute_entropy(log_stft_sample)
s_logmel_entropy = compute_entropy(s_logmel_sample)
# Print the results
print(f'STFT Sample Entropy: {stft_entropy:.4f}')
print(f'Log Mel Filter Bank Sample Entropy: {logmel_entropy:.4f}')
print(f'log STFT Sample Entropy: {log_stft_entropy:.4f}')
print(f's Log Mel Filter Bank Sample Entropy: {s_logmel_entropy:.4f}')


# Normalize the features
stft_sample_normalized = z_score_normalize(stft_sample)
logmel_sample_normalized = z_score_normalize(logmel_sample)
log_stft_sample_normalized = z_score_normalize(log_stft_sample)
s_logmel_sample_normalized = z_score_normalize(s_logmel_sample)
# Compute entropy
stft_entropy_norm = compute_entropy(stft_sample_normalized)
logmel_entropy_norm = compute_entropy(logmel_sample_normalized)
log_stft_entropy_norm = compute_entropy(stft_sample_normalized)
s_logmel_entropy_norm = compute_entropy(logmel_sample_normalized)
# Print the results
print(f'\nnormalzied STFT Sample Entropy: {stft_entropy_norm:.4f}')
print(f'normalized Log Mel Filter Bank Sample Entropy: {logmel_entropy_norm:.4f}')
print(f'normalzied log STFT Sample Entropy: {log_stft_entropy_norm:.4f}')
print(f'normalized s Log Mel Filter Bank Sample Entropy: {s_logmel_entropy_norm:.4f}')





# Compute histograms
stft_hist, stft_bin_edges = compute_histogram(stft_sample)
logmel_hist, logmel_bin_edges = compute_histogram(logmel_sample)
log_stft_hist, log_stft_bin_edges = compute_histogram(log_stft_sample)
s_logmel_hist, s_logmel_bin_edges = compute_histogram(s_logmel_sample)

# Plot Probability Distribution of STFT Sample
plt.figure(figsize=(8, 6))
plt.plot(stft_bin_edges[:-1], stft_hist, label='STFT', color='blue')
plt.fill_between(stft_bin_edges[:-1], stft_hist, alpha=0.3, color='blue')
plt.title('Probability Distribution of STFT Feature Sample', fontsize=16)
plt.xlabel('Feature Values', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.savefig('features/stft_prob_distribution.png', dpi=300)
plt.close()
# Plot Probability Distribution of Log Mel Filter Bank Sample
plt.figure(figsize=(8, 6))
plt.plot(logmel_bin_edges[:-1], logmel_hist, label='Log Mel Filter Bank', color='orange')
plt.fill_between(logmel_bin_edges[:-1], logmel_hist, alpha=0.3, color='orange')
plt.title('Probability Distribution of Log Mel Filter Bank Feature Sample', fontsize=16)
plt.xlabel('Feature Values', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.savefig('features/logmel_prob_distribution.png', dpi=300)
plt.close()
# Plot Probability Distribution of log STFT Sample
plt.figure(figsize=(8, 6))
plt.plot(log_stft_bin_edges[:-1], log_stft_hist, label='log STFT', color='blue')
plt.fill_between(log_stft_bin_edges[:-1], log_stft_hist, alpha=0.3, color='blue')
plt.title('Probability Distribution of log STFT Feature Sample', fontsize=16)
plt.xlabel('Feature Values', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.savefig('features/log_stft_prob_distribution.png', dpi=300)
plt.close()
# Plot Probability Distribution of s Log Mel Filter Bank Sample
plt.figure(figsize=(8, 6))
plt.plot(s_logmel_bin_edges[:-1], s_logmel_hist, label='s Log Mel Filter Bank', color='orange')
plt.fill_between(s_logmel_bin_edges[:-1], s_logmel_hist, alpha=0.3, color='orange')
plt.title('Probability Distribution of s Log Mel Filter Bank Feature Sample', fontsize=16)
plt.xlabel('Feature Values', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.savefig('features/s_logmel_prob_distribution.png', dpi=300)
plt.close()



# Compute histograms
stft_hist, stft_bin_edges = compute_histogram(stft_sample_normalized)
logmel_hist, logmel_bin_edges = compute_histogram(logmel_sample_normalized)
log_stft_hist, log_stft_bin_edges = compute_histogram(log_stft_sample_normalized)
s_logmel_hist, s_logmel_bin_edges = compute_histogram(s_logmel_sample_normalized)

# Plot Probability Distribution of STFT Sample
plt.figure(figsize=(8, 6))
plt.plot(stft_bin_edges[:-1], stft_hist, label='STFT', color='blue')
plt.fill_between(stft_bin_edges[:-1], stft_hist, alpha=0.3, color='blue')
plt.title('Probability Distribution of normalized STFT Feature Sample', fontsize=16)
plt.xlabel('Feature Values', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.savefig('features/normalized_stft_prob_distribution.png', dpi=300)
plt.close()
# Plot Probability Distribution of Log Mel Filter Bank Sample
plt.figure(figsize=(8, 6))
plt.plot(logmel_bin_edges[:-1], logmel_hist, label='Log Mel Filter Bank', color='orange')
plt.fill_between(logmel_bin_edges[:-1], logmel_hist, alpha=0.3, color='orange')
plt.title('Probability Distribution of normalized Log Mel Filter Bank Feature Sample', fontsize=16)
plt.xlabel('Feature Values', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.savefig('features/normalized_logmel_prob_distribution.png', dpi=300)
plt.close()
# Plot Probability Distribution of log STFT Sample
plt.figure(figsize=(8, 6))
plt.plot(log_stft_bin_edges[:-1], log_stft_hist, label='log STFT', color='blue')
plt.fill_between(log_stft_bin_edges[:-1], log_stft_hist, alpha=0.3, color='blue')
plt.title('Probability Distribution of normalized log STFT Feature Sample', fontsize=16)
plt.xlabel('Feature Values', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.savefig('features/normalized_log_stft_prob_distribution.png', dpi=300)
plt.close()
# Plot Probability Distribution of s Log Mel Filter Bank Sample
plt.figure(figsize=(8, 6))
plt.plot(s_logmel_bin_edges[:-1], s_logmel_hist, label='s Log Mel Filter Bank', color='orange')
plt.fill_between(s_logmel_bin_edges[:-1], s_logmel_hist, alpha=0.3, color='orange')
plt.title('Probability Distribution of normalized s Log Mel Filter Bank Feature Sample', fontsize=16)
plt.xlabel('Feature Values', fontsize=14)
plt.ylabel('Probability Density', fontsize=14)
plt.grid(True)
plt.legend(fontsize=14)
plt.savefig('features/normalized_s_logmel_prob_distribution.png', dpi=300)
plt.close()





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
    return np.mean(entropies), np.std(entropies)

# Compute entropy for multiple samples
stft_entropy_multiple, stft_m_std = compute_entropy_multiple(stft_features)
logmel_entropy_multiple, logmel_m_std = compute_entropy_multiple(logmel_features)
log_stft_entropy_multiple, log_stft_m_std = compute_entropy_multiple(log_stft_features)
s_logmel_entropy_multiple, s_logmel_m_std = compute_entropy_multiple(s_logmel_features)
# Print the results
print(f'\nAverage STFT Entropy: {stft_entropy_multiple:.4f}, std: {stft_m_std}')
print(f'Average log STFT Entropy: {log_stft_entropy_multiple:.4f}, std: {log_stft_m_std}')
print(f'Average Log Mel Filter Bank Entropy: {logmel_entropy_multiple:.4f}, std: {logmel_m_std}')
print(f'Average s Log Mel Filter Bank Entropy: {s_logmel_entropy_multiple:.4f}, std: {s_logmel_m_std}')





def compute_batch_stats(batch_data):
    """
    Compute the mean and standard deviation over an entire batch of features.

    Args:
        batch_data (numpy.ndarray): The batch of features with shape (N, H, W),
                                    where N is the number of samples, H is the height, and W is the width.

    Returns:
        tuple: (mean, std) where mean and std are the computed mean and standard deviation for the batch.
    """
    # Flatten the batch data to compute global mean and std
    data_flat = batch_data.flatten()
    mean = np.mean(data_flat)
    std = np.std(data_flat)
    return mean, std

def normalize_batch(batch_data, mean, std):
    """
    Normalize a batch of features using the provided mean and standard deviation.

    Args:
        batch_data (numpy.ndarray): The batch of features to normalize.
        mean (float): The mean to use for normalization.
        std (float): The standard deviation to use for normalization.

    Returns:
        numpy.ndarray: The normalized batch of features.
    """
    normalized_batch = (batch_data - mean) / std
    return normalized_batch

# Compute mean and std over the entire batch
mean, std = compute_batch_stats(stft_features)
# Normalize the batch using the computed mean and std
normalized_stft_features = normalize_batch(stft_features, mean, std)
print(f"\nstft Mean: {mean:.4f}, Std: {std:.4f}")

# Compute mean and std over the entire batch
mean, std = compute_batch_stats(log_stft_features)
# Normalize the batch using the computed mean and std
normalized_log_stft_features = normalize_batch(log_stft_features, mean, std)
print(f"log stft Mean: {mean:.4f}, Std: {std:.4f}")

# Compute mean and std over the entire batch
mean, std = compute_batch_stats(logmel_features)
# Normalize the batch using the computed mean and std
normalized_logmel_features = normalize_batch(logmel_features, mean, std)
print(f"log mel Mean: {mean:.4f}, Std: {std:.4f}")

# Compute mean and std over the entire batch
mean, std = compute_batch_stats(s_logmel_features)
# Normalize the batch using the computed mean and std
normalized_s_logmel_features = normalize_batch(s_logmel_features, mean, std)
print(f"s log mel Mean: {mean:.4f}, Std: {std:.4f}")



# Compute entropy for multiple samples
stft_entropy_multiple, stft_m_std = compute_entropy_multiple(normalized_stft_features)
logmel_entropy_multiple, logmel_m_std = compute_entropy_multiple(normalized_logmel_features)
log_stft_entropy_multiple, log_stft_m_std = compute_entropy_multiple(normalized_log_stft_features)
s_logmel_entropy_multiple, s_logmel_m_std = compute_entropy_multiple(normalized_s_logmel_features)
# Print the results
print(f'\nAverage Normalized STFT Entropy: {stft_entropy_multiple:.4f}, std: {stft_m_std}')
print(f'Average Normalized log STFT Entropy: {log_stft_entropy_multiple:.4f}, std: {log_stft_m_std}')
print(f'Average Normalized Log Mel Filter Bank Entropy: {logmel_entropy_multiple:.4f}, std: {logmel_m_std}')
print(f'Average Normalized s Log Mel Filter Bank Entropy: {s_logmel_entropy_multiple:.4f}, std: {s_logmel_m_std}')
