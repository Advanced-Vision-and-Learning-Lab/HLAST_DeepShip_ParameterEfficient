import os
import torch
import numpy as np
import scipy.io.wavfile as wav
from nnAudio import features
from torchlibrosa.stft import Spectrogram, LogmelFilterBank

# Paths
base_dir = 'Datasets/DeepShip/Segments_5s_16000hz'
categories = ['Cargo', 'Passengership', 'Tanker', 'Tug']

# Parameters
sr = 16000
n_fft = 1024
hop_length = 512
n_mels = 64

# STFT module
stft_transform = features.STFT(n_fft=n_fft, hop_length=hop_length, sr=sr)

# Spectrogram module for Log Mel Filter Bank
spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)

# Log Mel Filter Bank module
mel_transform = LogmelFilterBank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr // 2, freeze_parameters=True)

# Function to extract features
def extract_features(file_path):
    sr, y = wav.read(file_path)
    
    # Convert audio to floating-point
    y = y.astype(np.float32)
    
    # Convert to torch tensor
    y_tensor = torch.tensor(y).unsqueeze(0)  # add batch dimension
    
    # STFT
    stft_feature = stft_transform(y_tensor).numpy()
    
    # Spectrogram for Log Mel Filter Bank
    spectrogram = spectrogram_transform(y_tensor)
    
    # Log Mel Filter Bank
    logmel_feature = mel_transform(spectrogram).numpy()
    
    return stft_feature.squeeze(0), logmel_feature.squeeze(0)

# Iterate through dataset and save features
features = {'STFT': [], 'LogMel': []}

for category in categories:
    category_dir = os.path.join(base_dir, category)
    for recording in os.listdir(category_dir):
        recording_dir = os.path.join(category_dir, recording)
        for file in os.listdir(recording_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(recording_dir, file)
                stft_feature, logmel_feature = extract_features(file_path)
                features['STFT'].append(stft_feature)
                features['LogMel'].append(logmel_feature)

# Save features to disk
np.save('tb_logs/stft_features.npy', np.array(features['STFT']))
np.save('tb_logs/logmel_features.npy', np.array(features['LogMel']))
