import os
import random
import numpy as np
from scipy.io import wavfile
from torch.utils.data import Dataset, DataLoader
import lightning as L

# Mapping of folder names to class labels (as per your instructions)
CLASS_MAPPING = {
    'Fishboat': 'A', 'Trawler': 'A', 'Musselboat': 'A', 'Tugboat': 'A', 'Dagat': 'A',
    'Motorboat': 'B', 'Pilotship': 'B', 'Sailboat': 'B',
    'Passenger': 'C',
    'Oceanliner': 'D', 'RORO': 'D',
    'Naturalambientnoise': 'E'
}

CLASS_LABELS = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4
}

SPLITS_FILE = "shipsEar_splits.txt"

class ShipsEarSegmentedDataset(Dataset):
    def __init__(self, filepaths, labels):
        """
        Args:
            filepaths (list): List of file paths to .wav files.
            labels (list): List of corresponding labels for each file.
        """
        self.filepaths = filepaths
        self.labels = labels

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Get the file path and corresponding label
        wav_file = self.filepaths[idx]
        label = self.labels[idx]

        # Read the wav file using scipy.io.wavfile.read
        sample_rate, data = wavfile.read(wav_file)

        # Return the waveform and label as a tuple (no normalization applied here)
        return data.astype(np.float32), label


class ShipsEarDataModule(L.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        """Split dataset into train/val/test."""
        
        # Collect all file paths and labels from the dataset directory
        all_filepaths, all_labels = self._collect_files_and_labels(self.data_dir)

        # Check if splits file exists
        if os.path.exists(SPLITS_FILE):
            print(f"Loading splits from {SPLITS_FILE}")
            train_files, val_files, test_files = self.load_splits(SPLITS_FILE)
        
        else:
            print(f"{SPLITS_FILE} not found. Generating new splits.")
            # Generate new splits (70/10/20)
            all_files = list(zip(all_filepaths, all_labels))
            random.shuffle(all_files)

            train_size = int(0.7 * len(all_files))
            val_size = int(0.1 * len(all_files))
            test_size = len(all_files) - train_size - val_size

            train_files = all_files[:train_size]
            val_files = all_files[train_size:train_size + val_size]
            test_files = all_files[train_size + val_size:]

            # Save splits to file for future use
            self.save_splits(SPLITS_FILE, train_files, val_files, test_files)

        # Separate file paths and labels for each split
        train_filepaths, train_labels = zip(*train_files)
        val_filepaths, val_labels = zip(*val_files)
        test_filepaths, test_labels = zip(*test_files)

        # Create datasets for each split manually
        self.train_dataset = ShipsEarSegmentedDataset(list(train_filepaths), list(train_labels))
        self.val_dataset = ShipsEarSegmentedDataset(list(val_filepaths), list(val_labels))
        self.test_dataset = ShipsEarSegmentedDataset(list(test_filepaths), list(test_labels))

        # Print dataset statistics (only once when setup is called)
        print(f"Number of training samples: {len(self.train_dataset)}")
        print(f"Number of validation samples: {len(self.val_dataset)}")
        print(f"Number of test samples: {len(self.test_dataset)}")

    def _collect_files_and_labels(self, root_dir):
        """Collect all .wav files and their corresponding labels."""
        filepaths = []
        labels = []

        # Walk through each folder in root_dir and collect file paths and labels based on folder name
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path) and folder_name in CLASS_MAPPING:
                class_label = CLASS_LABELS[CLASS_MAPPING[folder_name]]
                for filename in os.listdir(folder_path):
                    if filename.endswith('.wav'):
                        file_path = os.path.join(folder_path, filename)
                        filepaths.append(file_path)
                        labels.append(class_label)

        return filepaths, labels

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def save_splits(self, filepath, train_files, val_files, test_files):
        """Save train/val/test splits to a text file."""
        with open(filepath, "w") as f:
            f.write("Train Files:\n")
            for filepath, label in train_files:
                f.write(f"{filepath},{label}\n")
            
            f.write("Validation Files:\n")
            for filepath, label in val_files:
                f.write(f"{filepath},{label}\n")
            
            f.write("Test Files:\n")
            for filepath, label in test_files:
                f.write(f"{filepath},{label}\n")

    def load_splits(self, filepath):
        """Load train/val/test splits from a text file."""
        with open(filepath) as f:
            lines = f.readlines()

            train_start_idx = lines.index("Train Files:\n") + 1
            val_start_idx = lines.index("Validation Files:\n") + 1
            test_start_idx = lines.index("Test Files:\n") + 1

            train_lines = lines[train_start_idx:val_start_idx - 1]
            val_lines = lines[val_start_idx:test_start_idx - 1]
            test_lines = lines[test_start_idx:]

            train_files = [self._parse_line(line) for line in train_lines]
            val_files = [self._parse_line(line) for line in val_lines]
            test_files = [self._parse_line(line) for line in test_lines]

        return train_files, val_files, test_files

    def _parse_line(self, line):
        """Parse a line from shipsEar_splits.txt to extract filepath and label."""
        filepath, label_str = line.strip().split(',')
        return filepath.strip(), int(label_str)


