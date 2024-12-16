import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import lightning as L
from scipy.io import wavfile

class ShipsEarDataset(Dataset):
    def __init__(self, segment_list):
        self.segment_list = segment_list

    def __len__(self):
        return len(self.segment_list)

    def __getitem__(self, idx):
        file_path, label = self.segment_list[idx]
        
        # Load the audio signal
        sample_rate, signal = wavfile.read(file_path)
        signal = torch.tensor(signal, dtype=torch.float)
        
        label = torch.tensor(label, dtype=torch.long)

        return signal, label

class ShipsEarDataModule(L.LightningDataModule):
    def __init__(self, parent_folder='./Datasets/ShipsEar', batch_size=None, num_workers=8,
                 train_split=0.7, val_test_split=1/3, random_seed=42, shuffle=False,
                 split_file='shipsear_data_split.txt'):
        super().__init__()
        
        self.batch_size = batch_size or {'train': 64, 'val': 128, 'test': 128}

        self.parent_folder = parent_folder
        self.train_split = train_split
        self.val_test_split = val_test_split
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.shuffle = shuffle
        
        # File to save/load splits
        self.split_file = split_file
        
        # Placeholder for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def save_splits(self, folder_lists):
        """Save train/val/test splits to a text file."""
        with open(self.split_file, 'w') as f:
            for split in ['train', 'val', 'test']:
                f.write(f"{split}:\n")
                for folder_path, label in folder_lists[split]:
                    f.write(f"{folder_path},{label}\n")
                    
    def load_splits(self):
        """Load train/val/test splits from a text file."""
        folder_lists = {'train': [], 'val': [], 'test': []}
        
        if not os.path.exists(self.split_file):
            raise FileNotFoundError(f"Split file {self.split_file} does not exist!")
        
        with open(self.split_file, 'r') as f:
            current_split = None
            for line in f:
                line = line.strip()
                if line.endswith(':'):
                    current_split = line[:-1]
                else:
                    folder_path, label = line.split(',')
                    folder_lists[current_split].append((folder_path, int(label)))
                    
        return folder_lists

    def check_data_leakage(self):
        """
        Checks for data leakage by ensuring that:
        1. No recording (subfolder) appears in more than one split (train, val, test).
        2. No segment (file) is duplicated across splits.
        """
        try:
            folder_lists = self.load_splits()
        except FileNotFoundError:
            print("Split file not found. Cannot check data leakage.")
            return

        splits = ['train', 'val', 'test']
        recordings = {split: set() for split in splits}
        segments = {split: set() for split in splits}

        # Collect all folder paths and segment file names for each split
        for split in splits:
            for folder_path, _ in folder_lists[split]:
                recordings[split].add(folder_path)
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith('.wav'):
                            segments[split].add(file)  # Assuming unique filenames

        # **Recording-Level Check**
        all_recordings = []
        for split in splits:
            all_recordings.extend(recordings[split])
        if len(all_recordings) != len(set(all_recordings)):
            print("Data leakage detected at the RECORDING level.")
            return
        else:
            print("No data leakage detected at the RECORDING level.")

        # **Segment-Level Check**
        all_segments = []
        for split in splits:
            all_segments.extend(segments[split])
        if len(all_segments) != len(set(all_segments)):
            print("Data leakage detected at the SEGMENT level.")
        else:
            print("No data leakage detected at the SEGMENT level.")

       
    def setup(self, stage=None):
        # Check if split file exists and load it if available
        if os.path.exists(self.split_file):
            print(f"\nLoading splits from {self.split_file}")
            folder_lists = self.load_splits()
        
        else:
            print(f"Creating new splits and saving them to {self.split_file}")
            
            # Read metadata file (shipsEar.xlsx)
            metadata_path = os.path.join(self.parent_folder, 'shipsEar.xlsx')
            metadata = pd.read_excel(metadata_path)

            # Get classes in directory (A, B, C, D, E)
            ships_classes = [f.name for f in os.scandir(self.parent_folder) if f.is_dir()]

            class_mapping = {ship: idx for idx, ship in enumerate(ships_classes)}

            folder_lists = {'train': [], 'test': [], 'val': []}

            # Loop over each class and split data into train/val/test sets at recording level (subfolder level)
            for label in ships_classes:
                label_path = os.path.join(self.parent_folder, label)
                subfolders = os.listdir(label_path)

                # Split subfolders into training and test/validation sets
                subfolders_train, subfolders_test_val = train_test_split(
                    subfolders,
                    train_size=self.train_split,
                    shuffle=self.shuffle,
                    random_state=self.random_seed,
                )

                # Split test/validation set further into test and validation sets
                subfolders_test, subfolders_val = train_test_split(
                    subfolders_test_val,
                    test_size=self.val_test_split,
                    shuffle=self.shuffle,
                    random_state=self.random_seed,
                )

                # Add subfolders to appropriate folder list with their class labels
                for subfolder in subfolders_train:
                    folder_lists['train'].append((os.path.join(label_path, subfolder), class_mapping[label]))

                for subfolder in subfolders_test:
                    folder_lists['test'].append((os.path.join(label_path, subfolder), class_mapping[label]))

                for subfolder in subfolders_val:
                    folder_lists['val'].append((os.path.join(label_path, subfolder), class_mapping[label]))
            
            # Save splits to a text file for future use.
            self.save_splits(folder_lists)

        segment_lists = {'train': [], 'test': [], 'val': []}

        # Loop over each partition and gather all segments (files) within each folder
        for split in ['train', 'test', 'val']:
            for folder_path, label in folder_lists[split]:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith('.wav'):
                            file_path = os.path.join(root, file)
                            segment_lists[split].append((file_path, label))
        
        # Initialize datasets
        self.train_dataset = ShipsEarDataset(segment_lists['train'])
        self.val_dataset = ShipsEarDataset(segment_lists['val'])
        self.test_dataset = ShipsEarDataset(segment_lists['test'])

        print(f"\nNumber of training samples: {len(self.train_dataset)}")
        print(f"Number of validation samples: {len(self.val_dataset)}")
        print(f"Number of test samples: {len(self.test_dataset)}\n")
        
        self.check_data_leakage()

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size['train'],
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size['val'],
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size['test'],
            num_workers=self.num_workers
        )
