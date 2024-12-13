#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 15:22:15 2024

@author: amir.m
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from scipy.io import wavfile
import lightning as L
import librosa
import random

class SSAudioDataset(Dataset):
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


class SSAudioDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, sample_rate, num_workers, test_size=0.2, val_size=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.class_to_idx = self.create_class_index_mapping()
        self.num_workers = num_workers
        self.prepared = False
        self.sample_rate = sample_rate
        self.raw_data_list = [] 

    def create_class_index_mapping(self):
        class_names = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        class_to_idx = {class_name: i for i, class_name in enumerate(sorted(class_names))}
        print(f"Class: {class_to_idx}")
        return class_to_idx

    def list_wav_files(self):
        wav_files = []
        for class_name in os.listdir(self.data_dir):
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.isdir(class_path):
                for recording in os.listdir(class_path):
                    recording_path = os.path.join(class_path, recording)
                    if os.path.isdir(recording_path):
                        for segment in os.listdir(recording_path):
                            if segment.endswith('.wav'):
                                segment_path = os.path.join(recording_path, segment)
                                wav_files.append(segment_path)
        print(f'Found {len(wav_files)} .wav files')
        return wav_files

    def read_wav_files(self, wav_files):
        data_list = []
        for file_path in wav_files:
            sampling_rate, data = wavfile.read(file_path)
            file_data = {
                'file_path': file_path,
                'sampling_rate': sampling_rate,
                'data': data
            }
            data_list.append(file_data)
        print(f'Read {len(data_list)} .wav files')
        self.raw_data_list = data_list  # Save the raw data list for later access
        return data_list
    
    def get_raw_audio_data(self):
        if self.raw_data_list:
            # Return the first raw audio data from the list
            return self.raw_data_list[0]['data']
        else:
            print("No raw audio data available.")
            return None

    def organize_data(self, data_list):
        organized_data = defaultdict(lambda: defaultdict(list))
        for file_data in data_list:
            path_parts = file_data['file_path'].split(os.sep)
            class_name = path_parts[-3]
            recording_name = path_parts[-2]
            organized_data[class_name][recording_name].append(file_data)
        print(f'Organized data into {len(organized_data)} classes')
        return organized_data


    def create_splits(self, organized_data):
        all_recordings = []
        for class_name, recordings in organized_data.items():
            for recording_name in recordings.keys():
                all_recordings.append((class_name, recording_name, organized_data[class_name][recording_name]))

        # Shuffling to ensure randomness
        random.seed(42)
        random.shuffle(all_recordings)

        # Calculating split indices
        total_recordings = len(all_recordings)
        num_test = int(total_recordings * self.test_size)
        num_val = int(total_recordings * self.val_size)
        num_train = total_recordings - num_test - num_val

        # Allocating recordings to splits
        test_recordings = all_recordings[:num_test]
        val_recordings = all_recordings[num_test:num_test + num_val]
        train_recordings = all_recordings[num_test + num_val:]

        # Extracting the actual data from the tuples
        train_data = [data for _, _, recordings in train_recordings for data in recordings]
        val_data = [data for _, _, recordings in val_recordings for data in recordings]
        test_data = [data for _, _, recordings in test_recordings for data in recordings]

        print('Created train, validation, and test splits')
        return train_data, val_data, test_data


    def check_data_leakage(self):
        print("\nChecking data leakage")

        all_data = self.train_data + self.val_data + self.test_data
        flattened_data = [item for sublist in all_data for item in (sublist if isinstance(sublist, list) else [sublist])]

        # Ensure flattened_data is a list of dictionaries with 'file_path' key
        if not isinstance(flattened_data, list):
            raise ValueError("flattened_data should be a list")
        if not all(isinstance(file_data, dict) for file_data in flattened_data):
            raise ValueError("Each element in flattened_data should be a dictionary")
        if not all('file_path' in file_data for file_data in flattened_data):
            raise ValueError("Each dictionary in flattened_data should contain the 'file_path' key")

        file_paths = [file_data['file_path'] for file_data in flattened_data]
        unique_file_paths = set(file_paths)

        if len(file_paths) != len(unique_file_paths):
            print("\nData leakage detected: Some samples are present in more than one split!\n")

            # Identify and print the duplicated file paths
            from collections import Counter
            file_path_counts = Counter(file_paths)
            duplicated_paths = [file_path for file_path, count in file_path_counts.items() if count > 1]

            print("\nDuplicated file paths:")
            for path in duplicated_paths:
                print(path)
        else:
            print("\nNo data leakage detected.\n")

    def save_split_indices(self, filepath):
        print("\nSaving split indices...")
        with open(filepath, 'w') as f:
            f.write('Train indices and paths:\n')
            for idx, file_data in enumerate(self.train_data):
                f.write(f'{idx}: {file_data["file_path"]}\n')
    
            f.write('\nValidation indices and paths:\n')
            for idx, file_data in enumerate(self.val_data):
                f.write(f'{idx}: {file_data["file_path"]}\n')
    
            f.write('\nTest indices and paths:\n')
            for idx, file_data in enumerate(self.test_data):
                f.write(f'{idx}: {file_data["file_path"]}\n')


    def load_split_indices(self, filepath, t_rate):
        print("\nLoading split indices from the saved file...\n")
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.raw_data_list = [] 

        first_file = True  
        current_split = None
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('Train indices and paths:'):
                    current_split = 'train'
                elif line.startswith('Validation indices and paths:'):
                    current_split = 'val'
                elif line.startswith('Test indices and paths:'):
                    current_split = 'test'
                elif line and not line.startswith('Train indices and paths:') and not line.startswith('Validation indices and paths:') and not line.startswith('Test indices and paths:'):
                    if current_split:
                        idx, file_path = line.split(': ', 1)
                        # Adjust the file path to include the sampling rate
                        parts = file_path.split('/')
                        #parts[3] = f'Segments_5s_{t_rate}hz'  # Adjust the directory to reflect the target sampling rate
                        adjusted_file_path = '/'.join(parts)
                        sampling_rate, data = wavfile.read(adjusted_file_path)
                        
                        file_data = {
                            'file_path': adjusted_file_path,
                            'sampling_rate': sampling_rate,
                            'data': data
                        }
                        
                        self.raw_data_list.append(file_data)  
                        
                        if current_split == 'train':
                            self.train_data.append(file_data)
                        elif current_split == 'val':
                            self.val_data.append(file_data)
                        elif current_split == 'test':
                            self.test_data.append(file_data)

        #if not self.prepared:
        self.check_data_leakage()
        self.get_raw_audio_data()

        self.prepared = True



    def prepare_data(self):
        split_indices_path = 'split_indices.txt'

        if os.path.exists(split_indices_path):
            if not self.prepared:  # Check if already prepared to avoid redundant loading
                self.load_split_indices(split_indices_path, t_rate=self.sample_rate)
                self.prepared = True                      
        else:
            if not self.prepared:
                self.wav_files = self.list_wav_files()
                self.data_list = self.read_wav_files(self.wav_files)
                self.organized_data = self.organize_data(self.data_list)
                self.train_data, self.val_data, self.test_data = self.create_splits(self.organized_data)
                
                self.check_data_leakage()
                
                self.save_split_indices(split_indices_path)  
                
                self.prepared = True

    
    def setup(self, stage=None):
        pass


    def train_dataloader(self):
        train_dataset = SSAudioDataset(self.train_data, self.class_to_idx)
        return DataLoader(train_dataset, batch_size=self.batch_size['train'], shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        val_dataset = SSAudioDataset(self.val_data, self.class_to_idx)
        return DataLoader(val_dataset, batch_size=self.batch_size['val'], shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        test_dataset = SSAudioDataset(self.test_data, self.class_to_idx)
        return DataLoader(test_dataset, batch_size=self.batch_size['test'], shuffle=False, num_workers=self.num_workers, pin_memory=True)
