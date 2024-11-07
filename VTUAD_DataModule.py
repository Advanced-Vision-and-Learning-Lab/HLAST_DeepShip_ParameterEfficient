import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from scipy.io import wavfile
import lightning as L

class AudioDataset(Dataset):
    def __init__(self, file_paths, class_mapping, global_min=None, global_max=None, normalize=False):
        self.file_paths = file_paths
        self.class_mapping = class_mapping
        self.global_min = global_min
        self.global_max = global_max
        self.normalize = normalize

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        
        # Extract the label from the folder name (e.g., 'background', 'cargo', etc.)
        label_str = os.path.basename(os.path.dirname(file_path))
        
        # Map the label string to its corresponding index
        label = self.class_mapping[label_str]

        # Load the audio file
        sample_rate, data = wavfile.read(file_path)
        data = data.astype(np.float32)

        # Normalize if required
        if self.normalize and self.global_min is not None and self.global_max is not None:
            data = (data - self.global_min) / (self.global_max - self.global_min)

        # Return both data and its corresponding label index
        return torch.tensor(data), torch.tensor(label)
    
class AudioDataModule(L.LightningDataModule):
    def __init__(self, base_dir, scenario_name, batch_size=32):
        super().__init__()
        self.base_dir = base_dir
        self.scenario_name = scenario_name
        self.batch_size = batch_size
        
        # Define the class mapping
        self.class_mapping = {'background': 0, 'cargo': 1, 'passengership': 2, 'tanker': 3, 'tug': 4}

    def setup(self, stage=None):
        # Define paths for train, validation, test sets
        scenario_path = os.path.join(self.base_dir, self.scenario_name)
        
        self.train_files = self._get_wav_files(os.path.join(scenario_path, 'train'))
        self.val_files = self._get_wav_files(os.path.join(scenario_path, 'validation'))
        self.test_files = self._get_wav_files(os.path.join(scenario_path, 'test'))

        # Compute global min and max from training data only
        global_min, global_max = self._compute_global_min_max(self.train_files)

        # Create datasets with normalization applied using global min/max from training set
        self.train_data = AudioDataset(self.train_files, class_mapping=self.class_mapping,
                                       global_min=global_min, global_max=global_max, normalize=True)
                                       
        self.val_data = AudioDataset(self.val_files, class_mapping=self.class_mapping,
                                     global_min=global_min, global_max=global_max, normalize=True)
                                     
        self.test_data = AudioDataset(self.test_files, class_mapping=self.class_mapping,
                                      global_min=global_min, global_max=global_max, normalize=True)

        # print the number of samples in each dataset split
        print(f"Number of training samples: {len(self.train_data)}")
        print(f"Number of validation samples: {len(self.val_data)}")
        print(f"Number of test samples: {len(self.test_data)}")
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, num_workers=8)


    def _get_wav_files(self, folder):
        """Helper function to get all .wav files in a folder."""
        wav_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        return wav_files

    def _compute_global_min_max(self, file_paths):
        """Efficiently compute the global min and max values across all training files."""
        global_min = float('inf')
        global_max = float('-inf')

        for file_path in file_paths:
            _, data = wavfile.read(file_path)
            data = data.astype(np.float32)
            file_min = np.min(data)
            file_max = np.max(data)

            if file_min < global_min:
                global_min = file_min
            if file_max > global_max:
                global_max = file_max

        return global_min, global_max

