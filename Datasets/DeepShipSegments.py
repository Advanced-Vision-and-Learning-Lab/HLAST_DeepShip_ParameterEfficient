# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 19:15:42 2023
Code modified from: https://github.com/lucascesarfd/underwater_snd/blob/master/nauta/one_stage/dataset.py
@author: jpeeples
"""
import pdb
import torch
import os
from scipy.io import wavfile
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torchaudio

# class DeepShipSegments(Dataset):
#     def __init__(self, parent_folder, train_split=.7,val_test_split=.5,
#                  partition='train', random_seed= 42, shuffle = False, transform=None, 
#                  target_transform=None):
#         self.parent_folder = parent_folder
#         self.folder_lists = {
#             'train': [],
#             'test': [],
#             'val': []
#         }
#         self.train_split = train_split
#         self.val_test_split = val_test_split
#         self.partition = partition
#         self.transform = transform
#         self.shuffle = shuffle
#         self.target_transform = target_transform
#         self.random_seed = random_seed
#         self.norm_function = None
#         self.class_mapping = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}

#         # Loop over each label and subfolder
#         for label in ['Cargo', 'Passengership', 'Tanker', 'Tug']:
#             label_path = os.path.join(parent_folder, label)
#             subfolders = os.listdir(label_path)
            
#             # Split subfolders into training, testing, and validation sets
#             subfolders_train, subfolders_test_val = train_test_split(subfolders, 
#                                                                      train_size=train_split, 
#                                                                      shuffle=self.shuffle, 
#                                                                      random_state=self.random_seed)
#             subfolders_test, subfolders_val = train_test_split(subfolders_test_val, 
#                                                                train_size=self.val_test_split, 
#                                                                shuffle=self.shuffle, 
#                                                                random_state=self.random_seed)

#             # Add subfolders to appropriate folder list
#             for subfolder in subfolders_train:
#                 subfolder_path = os.path.join(label_path, subfolder)
#                 self.folder_lists['train'].append((subfolder_path, self.class_mapping[label]))

#             for subfolder in subfolders_test:
#                 subfolder_path = os.path.join(label_path, subfolder)
#                 self.folder_lists['test'].append((subfolder_path, self.class_mapping[label]))

#             for subfolder in subfolders_val:
#                 subfolder_path = os.path.join(label_path, subfolder)
#                 self.folder_lists['val'].append((subfolder_path, self.class_mapping[label]))

#         self.segment_lists = {
#             'train': [],
#             'test': [],
#             'val': []
#         }

#         # Loop over each folder list and add corresponding files to file list
#         for split in ['train', 'test', 'val']:
#             for folder in self.folder_lists[split]:
#                 for root, dirs, files in os.walk(folder[0]):
#                     for file in files:
#                         if file.endswith('.wav'):
#                             file_path = os.path.join(root, file)
#                             label = folder[1]
#                             self.segment_lists[split].append((file_path, label))

#     def __len__(self):
#         return len(self.segment_lists[self.partition])

#     def __getitem__(self, idx):
#         file_path, label = self.segment_lists[self.partition][idx]    
        
        
        
#         sr, signal = wavfile.read(file_path, mmap=False)
#         signal = signal.astype(np.float32)
        

#         # Perform min-max normalization
#         if self.norm_function is not None:
#             signal = self.norm_function(signal)
#             signal = torch.tensor(signal)

        
#         label = torch.tensor(label)
#         if self.target_transform:
#             label = self.target_transform(label)

#         return signal, label, idx
    
    
    
    
import os
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold  
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold

class DeepShipSegments(Dataset):
    def __init__(self, parent_folder, num_folds=5, fold_index=0, run_number=0, partition='train', subdirs=None):
        self.run_number = run_number
        self.partition = partition
        self.class_mapping = {'Cargo': 0, 'Passengership': 1, 'Tanker': 2, 'Tug': 3}
        self.parent_folder = parent_folder
        self.num_folds = num_folds
        self.fold_index = fold_index
        self.segments = []
        self.groups = []  
        self.directory_log = []  
        self.normalize = False 
        self.overall_min = None  
        self.overall_max = None  


    
        group_id = 0
        for label_dir in sorted(os.listdir(parent_folder)):
            label_path = os.path.join(parent_folder, label_dir)
            if not os.path.isdir(label_path):  
                continue  
            for subdir in sorted(os.listdir(label_path)):
                subdir_path = os.path.join(label_path, subdir)
                if not os.path.isdir(subdir_path): 
                    continue  
                files = [f for f in os.listdir(subdir_path) if f.endswith('.wav')]
                for file in files:
                    file_path = os.path.join(subdir_path, file)
                    self.segments.append((file_path, label_dir))
                    self.groups.append(group_id)  # Same group ID for all files in the same subdir
                group_id += 1  # Increment group ID for each unique directory


        # Now split the data
        self.train_indices, self.val_indices = self._split_indices()
        
        
        ###
        # Set correct partition indices
        if self.partition == 'train':
            self.indices = self.train_indices
        elif self.partition == 'val':
            self.indices = self.val_indices
        else:
            raise ValueError("Invalid partition type. Use 'train' or 'val'.")



    def _split_indices(self):
        groups_array = np.array(self.groups)
        gkf = GroupKFold(n_splits=self.num_folds)
        splits = list(gkf.split(X=np.zeros(len(self.segments)), y=None, groups=groups_array))
        train_indices, val_indices = splits[self.fold_index]
        return train_indices, val_indices

    
    ####

    def __len__(self):
        if self.partition == 'train':
            return len(self.train_indices)
        elif self.partition == 'val':
            return len(self.val_indices)
        else:
            raise ValueError("Invalid partition type. Use 'train' or 'val'.")

    # def __len__(self):
    #     return len(self.indices)



    def set_normalization(self, overall_min, overall_max):
        self.overall_min = overall_min
        self.overall_max = overall_max
        self.normalize = True  # Enable normalization
     


    ###
    
    def __getitem__(self, idx):
        if self.partition == 'train':
            actual_idx = self.train_indices[idx % len(self.train_indices)]  
        elif self.partition == 'val':
            actual_idx = self.val_indices[idx % len(self.val_indices)]  
        else:
            raise ValueError("Invalid partition type. Use 'train' or 'val'.")
      
    
    # def __getitem__(self, idx):
    #     actual_idx = self.indices[idx]
    
    
        file_path, label_name = self.segments[actual_idx]
        label = self.class_mapping[label_name]  # Convert label name to integer using class_mapping
    

        sr, signal = wavfile.read(file_path)
        signal = signal.astype(np.float32)
    
        # Normalize the signal if normalization parameters are set
        if self.normalize:
            signal = 2 * (signal - self.overall_min) / (self.overall_max - self.overall_min) - 1
    
        return torch.tensor(signal, dtype=torch.float), torch.tensor(label, dtype=torch.long)  
    
    
    
    def get_metadata(self):
        file_paths = []
        labels = []

        for idx in range(len(self.segments)):
            file_path, label_name = self.segments[idx]
            label = self.class_mapping[label_name]
            file_paths.append(file_path)
            labels.append(label)

        return file_paths, labels
    
    
        
