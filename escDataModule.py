# import os
# import lightning as L
# from torch.utils.data import Dataset, DataLoader
# import pandas as pd
# from scipy.io import wavfile
# import numpy as np

# class ESC50Dataset(Dataset):
#     def __init__(self, data_dir, file_list):
#         self.data_dir = data_dir
#         self.file_list = file_list

#     def __len__(self):
#         return len(self.file_list)
    
#     def __getitem__(self, idx):
#         row = self.file_list.iloc[idx]
#         audio_path = os.path.join(self.data_dir, 'audio', row['filename'])
#         _, waveform = wavfile.read(audio_path)
#         waveform = waveform.astype(np.float32)
#         label = row['target']
#         return waveform, label
    
# class ESC50DataModule(L.LightningDataModule):
#     def __init__(self, data_dir: str, batch_size: dict, num_workers: int = 8):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.fold = 1  

#     def setup(self, stage: str = None):
#         metadata = pd.read_csv(os.path.join(self.data_dir, 'meta', 'esc50.csv'))
#         self.metadata = metadata

#     def set_fold(self, fold: int):
#         assert 1 <= fold <= 5, "Fold should be between 1 and 5"
#         self.fold = fold
        
#     def train_dataloader(self):
#         train_data = self.metadata[self.metadata['fold'] != self.fold]
#         return DataLoader(ESC50Dataset(self.data_dir, train_data), 
#                           batch_size=self.batch_size['train'], 
#                           num_workers=self.num_workers, 
#                           shuffle=True)

#     def val_dataloader(self):
#         val_data = self.metadata[self.metadata['fold'] == self.fold]
#         return DataLoader(ESC50Dataset(self.data_dir, val_data), 
#                           batch_size=self.batch_size['val'], 
#                           num_workers=self.num_workers)
                          
import os
import lightning as L
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from scipy.io import wavfile
import numpy as np
import torch


class ESC50Dataset(Dataset):
    def __init__(self, data_dir, file_list, global_min, global_max):
        self.data_dir = data_dir
        self.file_list = file_list
        self.global_min = global_min
        self.global_max = global_max

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        row = self.file_list.iloc[idx]
        audio_path = os.path.join(self.data_dir, 'audio', row['filename'])
        _, waveform = wavfile.read(audio_path)
        waveform = waveform.astype(np.float32)
        
        # Apply normalization here
        waveform = (waveform - self.global_min) / (self.global_max - self.global_min + 1e-8)
        
        label = row['target']
        return torch.tensor(waveform), torch.tensor(label)

class ESC50DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: dict, num_workers: int = 8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = 1
        self.global_min = None
        self.global_max = None
        self.metadata = None
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: str = None):
        if self.metadata is None:
            self.metadata = pd.read_csv(os.path.join(self.data_dir, 'meta', 'esc50.csv'))
        self._compute_global_min_max()
        self._create_datasets()

    def set_fold(self, fold: int):
        assert 1 <= fold <= 5, "Fold should be between 1 and 5"
        self.fold = fold
        self._compute_global_min_max()
        self._create_datasets()
        print(f"Set fold to {self.fold}")

    def _compute_global_min_max(self):
        train_data = self.metadata[self.metadata['fold'] != self.fold]
        file_paths = [os.path.join(self.data_dir, 'audio', row['filename']) for _, row in train_data.iterrows()]
        
        global_min = float('inf')
        global_max = float('-inf')

        for file_path in file_paths:
            _, data = wavfile.read(file_path)
            data = data.astype(np.float32)
            file_min = np.min(data)
            file_max = np.max(data)

            global_min = min(global_min, file_min)
            global_max = max(global_max, file_max)

        self.global_min = global_min
        self.global_max = global_max
        print(f"Fold {self.fold}: Global min = {self.global_min}, Global max = {self.global_max}")

    def _create_datasets(self):
        train_data = self.metadata[self.metadata['fold'] != self.fold]
        val_data = self.metadata[self.metadata['fold'] == self.fold]
        
        self.train_dataset = ESC50Dataset(self.data_dir, train_data, self.global_min, self.global_max)
        self.val_dataset = ESC50Dataset(self.data_dir, val_data, self.global_min, self.global_max)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size['train'], 
                          num_workers=self.num_workers, 
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                          batch_size=self.batch_size['val'], 
                          num_workers=self.num_workers)
