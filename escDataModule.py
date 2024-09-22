import os
import lightning as L
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchaudio
from sklearn.model_selection import KFold

class ESC50Dataset(Dataset):
    def __init__(self, data_dir, file_list):
        self.data_dir = data_dir
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        row = self.file_list.iloc[idx]
        audio_path = os.path.join(self.data_dir, 'audio', row['filename'])
        waveform, sample_rate = torchaudio.load(audio_path)
        label = row['target']
        return waveform, label

class ESC50DataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: dict, num_workers: int = 16):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = 0
        

    def setup(self, stage: str = None):
        metadata = pd.read_csv(os.path.join(self.data_dir, 'meta', 'esc50.csv'))
        self.kf = KFold(n_splits=3, shuffle=True, random_state=42)
        self.folds = list(self.kf.split(metadata))
        self.metadata = metadata

    def set_fold(self, fold: int):
        assert 0 <= fold < 3, "Fold should be 0, 1, or 2"
        self.fold = fold
        
    def train_dataloader(self):
        train_data = self.metadata.iloc[self.folds[self.fold][0]]
        return DataLoader(ESC50Dataset(self.data_dir, train_data), 
                          batch_size=self.batch_size['train'], 
                          num_workers=self.num_workers, 
                          shuffle=True)

    def val_dataloader(self):
        val_data = self.metadata.iloc[self.folds[self.fold][1]]
        return DataLoader(ESC50Dataset(self.data_dir, val_data), 
                          batch_size=self.batch_size['val'], 
                          num_workers=self.num_workers)
    
