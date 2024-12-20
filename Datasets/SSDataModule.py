import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from scipy.io import wavfile
import lightning as L
import random

class SSAudioDataset(Dataset):
    def __init__(self, data_list, class_to_idx):
        self.data_list = data_list
        self.class_to_idx = class_to_idx
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        file_data = self.data_list[idx]
        file_path = file_data['file_path']
        class_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        label = self.class_to_idx[class_name]
        
        _, data = wavfile.read(file_path)
        data_tensor = torch.tensor(data, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return data_tensor, label_tensor


class SSAudioDataModule(L.LightningDataModule):
    def __init__(self, data_dir='./Datasets/DeepShip/Segments_5s_16000hz/', batch_size=None,
                 num_workers=8, test_size=0.2, val_size=0.1):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size or {'train': 64, 'val': 128, 'test': 128}
        self.test_size = test_size
        self.val_size = val_size
        self.class_to_idx = self.create_class_index_mapping()
        self.num_workers = num_workers
        self.prepared = False

    def create_class_index_mapping(self):
        class_names = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]
        class_to_idx = {class_name: i for i, class_name in enumerate(sorted(class_names))}
        print(f"Class: {class_to_idx}")
        return class_to_idx

    def list_wav_files(self):
        wav_files = [
            os.path.join(self.data_dir, class_name, recording, segment)
            for class_name in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, class_name))
            for recording in os.listdir(os.path.join(self.data_dir, class_name))
            if os.path.isdir(os.path.join(self.data_dir, class_name, recording))
            for segment in os.listdir(os.path.join(self.data_dir, class_name, recording))
            if segment.endswith('.wav')
        ]
        print(f'Found {len(wav_files)} .wav files')
        return wav_files

    def read_wav_files(self, wav_files):
        data_list = [{'file_path': fp} for fp in wav_files]
        print(f'Read {len(data_list)} .wav files')
        return data_list

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
    
        # all_data is a list of dictionaries with 'file_path' key
        if not isinstance(all_data, list):
            raise ValueError("all_data should be a list")
        if not all(isinstance(file_data, dict) for file_data in all_data):
            raise ValueError("Each element in all_data should be a dictionary")
        if not all('file_path' in file_data for file_data in all_data):
            raise ValueError("Each dictionary in all_data should contain the 'file_path' key")
    
        file_paths = [file_data['file_path'] for file_data in all_data]
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
            print("\nNo data leakage detected.")

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

    def load_split_indices(self, filepath):
        print("\nLoading split indices from the saved file...\n")
        self.train_data = []
        self.val_data = []
        self.test_data = []

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
                        adjusted_file_path = file_path  
                                                
                        file_data = {
                            'file_path': adjusted_file_path
                        }
                        
                        if current_split == 'train':
                            self.train_data.append(file_data)
                        elif current_split == 'val':
                            self.val_data.append(file_data)
                        elif current_split == 'test':
                            self.test_data.append(file_data)

        print(f"Number of training samples: {len(self.train_data)}")
        print(f"Number of validation samples: {len(self.val_data)}")
        print(f"Number of test samples: {len(self.test_data)}")

        self.check_data_leakage()
        self.prepared = True

    def prepare_data(self):
        split_indices_path = 'split_indices.txt'

        if os.path.exists(split_indices_path):
            if not self.prepared:  
                self.load_split_indices(split_indices_path)
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
