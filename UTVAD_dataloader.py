"""
Created on Tue Oct 25 10:22:20 2024

@author: jarin.ritu
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile
import lightning as L

# Data Loader for VTUAD
class DeepShipSegments(Dataset):
    def __init__(self, parent_folder, partition='train', transform=None, 
                  target_transform=None, norm_function=None, shuffle=True, random_seed=42):
        self.parent_folder = parent_folder
        self.partition = partition
        self.transform = transform
        self.target_transform = target_transform
        self.norm_function = norm_function
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.class_mapping = {'background': 0, 'cargo': 1, 'passengership': 2, 'tanker': 3, 'tug': 4}
        self.segment_list = []

        self._collect_segments()
        self.global_min, self.global_max = self.compute_global_min_max()
    
    def _collect_segments(self):
        partition_path = os.path.join(self.parent_folder, self.partition, 'audio')
        for label, class_idx in self.class_mapping.items():
            label_path = os.path.join(partition_path, label)
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Directory {label_path} does not exist.")
            for root, _, files in os.walk(label_path):
                for file in files:
                    if file.endswith('.wav'):
                        file_path = os.path.join(root, file)
                        self.segment_list.append((file_path, class_idx))

        if self.shuffle:
            np.random.seed(self.random_seed)
            np.random.shuffle(self.segment_list)

    def __len__(self):
        return len(self.segment_list)

    def __getitem__(self, idx):
        file_path, label = self.segment_list[idx]
        try:
            sr, signal = wavfile.read(file_path, mmap=False)
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {e}")

        signal = signal.astype(np.float32)
        if self.norm_function is not None:
            signal = self.norm_function(signal)
        signal = torch.tensor(signal)

        if self.target_transform:
            label = self.target_transform(torch.tensor(label))
        else:
            label = torch.tensor(label)

        return signal, label, idx

    def compute_global_min_max(self):
        global_min = float('inf')
        global_max = float('-inf')
        for file_path, _ in self.segment_list:
            try:
                sr, signal = wavfile.read(file_path, mmap=False)
                signal = signal.astype(np.float32)
                file_min = np.min(signal)
                file_max = np.max(signal)
                if file_min < global_min:
                    global_min = file_min
                if file_max > global_max:
                    global_max = file_max
            except Exception as e:
                raise RuntimeError(f"Error reading file {file_path}: {e}")
        return global_min, global_max

    def set_norm_function(self, global_min, global_max):
        self.norm_function = lambda x: (x - global_min) / (global_max - global_min)
        print(f"Normalization function set with global_min: {global_min}, global_max: {global_max}")

class DeepShipDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers=4, pin_memory=True, shuffle=True, random_seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.norm_function = None
        self.global_min = None
        self.global_max = None

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = DeepShipSegments(self.data_dir, partition='train', shuffle=self.shuffle)
            self.val_dataset = DeepShipSegments(self.data_dir, partition='validation', shuffle=False)
            self.global_min, self.global_max = self.train_dataset.compute_global_min_max()
            self.norm_function = lambda x: (x - self.global_min) / (self.global_max - self.global_min)
            self.train_dataset.set_norm_function(self.global_min, self.global_max)
            self.val_dataset.set_norm_function(self.global_min, self.global_max)

        if stage == 'test' or stage is None:
            self.test_dataset = DeepShipSegments(self.data_dir, partition='test', shuffle=False)
            if self.global_min is None or self.global_max is None:
                raise RuntimeError("Global min and max must be computed during the 'fit' stage before testing.")
            self.test_dataset.set_norm_function(self.global_min, self.global_max)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size['train'], shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size['val'], shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size['test'], shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)
