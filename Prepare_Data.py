#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:13:10 2024

@author: amir.m
"""

import os  # For directory operations
from scipy.io import wavfile  # For reading .wav files

# Function to list all .wav files in the dataset directory
def list_wav_files(data_dir):
    wav_files = []  # Initialize an empty list to store file paths
    
    # Traverse the directory structure
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)  # Get path of each class folder
        if os.path.isdir(class_path):
            for recording in os.listdir(class_path):
                recording_path = os.path.join(class_path, recording)  # Get path of each recording folder
                if os.path.isdir(recording_path):
                    for segment in os.listdir(recording_path):
                        if segment.endswith('.wav'):
                            segment_path = os.path.join(recording_path, segment)  # Get path of each .wav file
                            wav_files.append(segment_path)  # Append the file path to the list
    
    return wav_files  # Return the list of .wav file paths


# Function to read .wav files and extract their data
def read_wav_files(wav_files):
    data_list = []  # Initialize an empty list to store the data
    
    for file_path in wav_files:
        # Read the .wav file
        sampling_rate, data = wavfile.read(file_path)
        
        # Store the information in a dictionary
        file_data = {
            'file_path': file_path,  # Path to the .wav file
            'sampling_rate': sampling_rate,  # Sampling rate of the .wav file
            'data': data  # Audio data
        }
        
        # Append the dictionary to the list
        data_list.append(file_data)
    
    return data_list  # Return the list of dictionaries


from collections import defaultdict

# Function to organize data by class and recording
def organize_data(data_list):
    organized_data = defaultdict(lambda: defaultdict(list))  # Initialize a nested dictionary
    
    for file_data in data_list:
        # Extract the class name and recording name from the file path
        path_parts = file_data['file_path'].split(os.sep)
        class_name = path_parts[-3]  # Assuming the class folder is 3 levels up from the file
        recording_name = path_parts[-2]  # Assuming the recording folder is 2 levels up from the file
        
        # Add the file data to the appropriate class and recording
        organized_data[class_name][recording_name].append(file_data)
    
    return organized_data  # Return the organized data



import numpy as np

from sklearn.model_selection import StratifiedKFold

# Function to create stratified k-folds
def create_stratified_k_folds(organized_data, num_folds):
    all_recording_names = []  # List to hold all recording names and their classes
    class_labels = []  # List to hold corresponding class labels
    
    # Collect all recording names and their classes
    for class_name, recordings in organized_data.items():
        for recording_name in recordings.keys():
            all_recording_names.append((class_name, recording_name))
            class_labels.append(class_name)
    
    skf = StratifiedKFold(n_splits=num_folds)  # Initialize StratifiedKFold
    
    # Initialize lists to hold training and validation splits
    train_folds = []
    val_folds = []
    
    # Perform the stratified k-fold split
    for train_index, val_index in skf.split(all_recording_names, class_labels):
        train_data = []
        val_data = []
        
        for idx in train_index:
            class_name, recording_name = all_recording_names[idx]
            train_data.extend(organized_data[class_name][recording_name])
        
        for idx in val_index:
            class_name, recording_name = all_recording_names[idx]
            val_data.extend(organized_data[class_name][recording_name])
        
        train_folds.append(train_data)  # Append training data for the fold
        val_folds.append(val_data)  # Append validation data for the fold
    
    return train_folds, val_folds  # Return the training and validation folds

# Function to count samples from each class
def count_samples_per_class(data_list):
    class_counts = defaultdict(int)  # Dictionary to store counts
    for file_data in data_list:
        class_name = file_data['file_path'].split(os.sep)[-3]  # Extract class name from the file path
        class_counts[class_name] += 1  # Increment count for the class
    return class_counts



# Function to check for data leakage
def check_data_leakage(train_folds, val_folds, num_folds):
    # Initialize a dictionary to count the occurrences of each sample in training and validation sets
    sample_counts = defaultdict(lambda: {'train': 0, 'val': 0})

    # Iterate over each fold
    for i in range(num_folds):
        train_data = train_folds[i]
        val_data = val_folds[i]

        # Check for overlap between train and val sets in the current fold
        train_set = set(file_data['file_path'] for file_data in train_data)
        val_set = set(file_data['file_path'] for file_data in val_data)
        overlap = train_set.intersection(val_set)

        if overlap:
            print(f"\n\nData leakage detected in fold {i + 1}! Overlap samples: {len(overlap)}\n")

        # Update sample counts for training and validation sets
        for file_data in train_data:
            sample_counts[file_data['file_path']]['train'] += 1

        for file_data in val_data:
            sample_counts[file_data['file_path']]['val'] += 1

    # Check if each sample appears exactly once in validation and k-1 times in training
    for file_path, counts in sample_counts.items():
        if counts['val'] != 1 or counts['train'] != num_folds - 1:
            print(f"Sample {file_path} does not meet the expected distribution! Train count: {counts['train']}, Val count: {counts['val']}")


# Function to calculate the global min and max values from the training data
def get_min_max_train(train_data):
    # Initialize min and max values
    global_min = float('inf')
    global_max = float('-inf')
    
    # Iterate through each file in the training data
    for file_data in train_data:
        data = file_data['data']  # Get the audio data
        
        data = data.astype(np.float32)
        
        # Update global min and max values
        file_min = np.min(data)
        file_max = np.max(data)
        if file_min < global_min:
            global_min = file_min
        if file_max > global_max:
            global_max = file_max
    
    return global_min, global_max  # Return the global min and max values

# Function to normalize audio data using global min and max values
def normalize_data(data_list, global_min, global_max):
    normalized_data_list = []  # Initialize a list to store normalized data
    
    # Ensure global_min and global_max are properly handled to avoid overflow
    global_min = np.float32(global_min)
    global_max = np.float32(global_max)
    
    # Iterate through each file in the data list
    for file_data in data_list:
        data = file_data['data']  # Get the audio data
        
        # Ensure the data is in the proper format (e.g., int16)
        data = data.astype(np.float32)
        
        # Normalize the data
        normalized_data = (data - global_min) / (global_max - global_min)
        
        # Store the normalized data in a new dictionary
        normalized_file_data = {
            'file_path': file_data['file_path'],  # Path to the .wav file
            'sampling_rate': file_data['sampling_rate'],  # Sampling rate of the .wav file
            'data': normalized_data  # Normalized audio data
        }
        
        # Append the normalized file data to the list
        normalized_data_list.append(normalized_file_data)
    
    return normalized_data_list  # Return the list of normalized data


import torch
from torch.utils.data import Dataset, DataLoader

# Define the class-to-index mapping
class_to_idx = {
    'Cargo': 0,
    'Passengership': 1,
    'Tanker': 2,
    'Tug': 3
}
class AudioDataset(Dataset):
    def __init__(self, data_list, class_to_idx):
        self.data_list = data_list  # Store the list of data
        self.class_to_idx = class_to_idx  # Store the class-to-index mapping

    def __len__(self):
        return len(self.data_list)  # Return the number of samples

    def __getitem__(self, idx):
        file_data = self.data_list[idx]  # Get the data for the given index
        data = file_data['data']  # Extract the normalized audio data
        class_name = file_data['file_path'].split(os.sep)[-3]  # Extract the class name from the file path
        label = self.class_to_idx[class_name]  # Convert the class name to an integer label
        data_tensor = torch.tensor(data, dtype=torch.float32)  # Convert data to a PyTorch tensor
        label_tensor = torch.tensor(label, dtype=torch.long)  # Convert label to a PyTorch tensor
        return data_tensor, label_tensor  # Return the data and label tensors



# Function to prepare data loaders for a given fold
def Prepare_DataLoaders(Network_parameters, fold_index, train_folds, val_folds):

    data_dir = Network_parameters["data_dir"]  # Get the data directory from network parameters
    
    # Step 5: Get the training data for the current fold
    train_data = train_folds[fold_index]
    
    # Step 6: Calculate global min and max from the training data
    global_min, global_max = get_min_max_train(train_data)

    # Step 7: Normalize the training and validation data
    normalized_train_data = normalize_data(train_data, global_min, global_max)
    normalized_val_data = normalize_data(val_folds[fold_index], global_min, global_max)
    print(f'Normalized {len(normalized_train_data)} training files and {len(normalized_val_data)} validation files')  # Print the number of normalized files
    
    # Step 8: Create PyTorch datasets
    class_to_idx = {'Cargo': 0,'Passengership': 1,'Tanker': 2,'Tug': 3}
    train_dataset = AudioDataset(normalized_train_data, class_to_idx)  # Create a dataset for the training data
    val_dataset = AudioDataset(normalized_val_data, class_to_idx)  # Create a dataset for the validation data

    # Step 9: Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=8, pin_memory=True)
    print(f'Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches\n')  # Print the number of batches
    
    # Return the data loaders
    return {'train': train_loader, 'val': val_loader}


