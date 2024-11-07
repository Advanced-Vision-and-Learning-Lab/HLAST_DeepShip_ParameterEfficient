import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch
import lightning as L
from scipy.io import wavfile


class ShipsEarDataset(Dataset):
    def __init__(self, segment_list, transform=None, target_transform=None):
        self.segment_list = segment_list
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.segment_list)

    def __getitem__(self, idx):
        # Access both the file path and normalized signal
        file_path, signal, label = self.segment_list[idx]
        
        # No need to read from wavfile again since we already have the signal
        if self.transform:
            signal = self.transform(signal)
            
        label = torch.tensor(label, dtype=torch.long)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return signal, label


class ShipsEarDataModule(L.LightningDataModule):
    def __init__(self, parent_folder, train_split=0.7, val_test_split=1/3,
                 batch_size=32, num_workers=8, random_seed=42, shuffle=False,
                 transform=None, target_transform=None, split_file='data_split.txt'):
        super().__init__()
        
        self.parent_folder = parent_folder
        self.train_split = train_split
        self.val_test_split = val_test_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.shuffle = shuffle
        self.transform = transform
        self.target_transform = target_transform
        
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



    def get_min_max_train(self):
        """Compute global min and max from the training dataset."""
        global_min = float('inf')
        global_max = float('-inf')
        
        for file_path, _ in self.train_dataset.segment_list:
            sr, signal = wavfile.read(file_path)
            signal = signal.astype(np.float32)  # Ensure float32 for consistency
            
            file_min = np.min(signal)
            file_max = np.max(signal)
            
            if file_min < global_min:
                global_min = file_min
            if file_max > global_max:
                global_max = file_max
        
        return global_min, global_max

    def normalize_data(self, data_list):
        """Normalize a list of data using precomputed global min and max."""
        normalized_data_list = []
        
        for file_path, label in data_list:
            sr, signal = wavfile.read(file_path)  # This expects a valid file path
            signal = signal.astype(np.float32)  # Ensure float32 for consistency
            
            # Apply min-max normalization
            normalized_signal = (signal - self.global_min) / (self.global_max - self.global_min)
            
            # Keep both the original file path and normalized signal
            normalized_data_list.append((file_path, normalized_signal, label))  # Keep file_path intact
        
        return normalized_data_list

    def setup(self, stage=None):
        # Check if split file exists and load it if available
        if os.path.exists(self.split_file):
            print(f"Loading splits from {self.split_file}")
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

        # Create datasets for each partition using the segment lists
        self.train_dataset = ShipsEarDataset(segment_lists['train'], transform=self.transform,
                                             target_transform=self.target_transform)

        self.val_dataset = ShipsEarDataset(segment_lists['val'], transform=self.transform,
                                           target_transform=self.target_transform)

        self.test_dataset = ShipsEarDataset(segment_lists['test'], transform=self.transform,
                                            target_transform=self.target_transform)

        # Compute global min/max from training dataset after loading it.
        print("Computing Global Min/Max from Training Data...")
        self.global_min, self.global_max = self.get_min_max_train()

        print(f"Global Min: {self.global_min}, Global Max: {self.global_max}")

        # Normalize train/val/test datasets after computing global min/max.
        print("Normalizing datasets...")
        
        # Normalize each dataset using the computed global min/max values.
        self.train_dataset.segment_list = [(self.normalize_data([segment])[0]) for segment in segment_lists['train']]
        self.val_dataset.segment_list = [(self.normalize_data([segment])[0]) for segment in segment_lists['val']]
        self.test_dataset.segment_list = [(self.normalize_data([segment])[0]) for segment in segment_lists['test']]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          num_workers=self.num_workers)

if __name__ == '__main__':
    data_dir = './ShipsEar/'
    
    # Initialize the data module with default parameters or customized ones.
    data_module = ShipsEarDataModule(data_dir)
    
    # Setup the data (splitting and loading datasets)
    data_module.setup()

    # Print dataset sizes for verification
    print(f"Number of training samples: {len(data_module.train_dataset)}")
    print(f"Number of validation samples: {len(data_module.val_dataset)}")
    print(f"Number of test samples: {len(data_module.test_dataset)}")
