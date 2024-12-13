import os
import torch
from torch.utils.data import DataLoader, Dataset
from scipy.io import wavfile
import lightning as L

class AudioDataset(Dataset):
    def __init__(self, file_paths, class_mapping):
        """
        Initialize the dataset with file paths and a class mapping dictionary.
        """
        self.file_paths = file_paths
        self.class_mapping = class_mapping

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
        data = torch.tensor(data, dtype=torch.float32)

        return data, torch.tensor(label, dtype=torch.long)
    
class AudioDataModule(L.LightningDataModule):
    def __init__(self, base_dir, scenario_name, batch_size, num_workers):
        """
        Initialize the data module with base directory, scenario name, batch sizes, and number of workers.
        """
        super().__init__()
        self.base_dir = base_dir
        self.scenario_name = scenario_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Define the class mapping
        self.class_mapping = {
            'background': 0,
            'cargo': 1,
            'passengership': 2,
            'tanker': 3,
            'tug': 4
        }

    def setup(self, stage=None):
        """
        Setup datasets for training, validation, and testing.
        """
        # Define paths for train, validation, test sets
        scenario_path = os.path.join(self.base_dir, self.scenario_name)
        
        self.train_files = self._get_wav_files(os.path.join(scenario_path, 'train'))
        self.val_files = self._get_wav_files(os.path.join(scenario_path, 'validation'))
        self.test_files = self._get_wav_files(os.path.join(scenario_path, 'test'))

        # Create datasets without normalization
        self.train_data = AudioDataset(self.train_files, class_mapping=self.class_mapping)
                                       
        self.val_data = AudioDataset(self.val_files, class_mapping=self.class_mapping)
                                     
        self.test_data = AudioDataset(self.test_files, class_mapping=self.class_mapping)

        # Print the number of samples in each dataset split
        print(f"Number of training samples: {len(self.train_data)}")
        print(f"Number of validation samples: {len(self.val_data)}")
        print(f"Number of test samples: {len(self.test_data)}")
        
    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size['train'],
            num_workers=self.num_workers,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size['val'],
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size['test'],
            num_workers=self.num_workers
        )

    def _get_wav_files(self, folder):
        """Helper function to get all .wav files in a folder."""
        wav_files = []
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        return wav_files
