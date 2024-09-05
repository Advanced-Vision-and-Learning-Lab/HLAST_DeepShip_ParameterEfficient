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

hop_length_stft = 1024  
hop_length_mel = 160  
n_mels = 64
freq_bins = 48  

# STFT module
stft_transform = features.STFT(sr=sr, n_fft=n_fft, hop_length=hop_length_stft, fmin=1e-6, fmax=sr // 2,
                               win_length=n_fft, freq_bins=48, output_format='Magnitude',trainable=False)

# log STFT module
log_stft_transform = features.STFT(sr=sr, n_fft=n_fft, hop_length=hop_length_stft, fmin=1e-6, fmax=sr // 2,
                                   win_length=n_fft, freq_bins=48, freq_scale='log2', output_format='Magnitude',trainable=False)

# Spectrogram module for Log Mel Filter Bank
spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length_mel,
                                    win_length=n_fft, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)

# Log Mel Filter Bank module
mel_transform = LogmelFilterBank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr // 2, freeze_parameters=True)


n_fft = 1024
hop_length_mel = 1000  
n_mels = 48
# Spectrogram module for Log Mel Filter Bank
s_spectrogram_transform = Spectrogram(n_fft=n_fft, hop_length=hop_length_mel,
                                    win_length=n_fft, window='hann', center=True, pad_mode='reflect', freeze_parameters=True)
# Log Mel Filter Bank module
s_mel_transform = LogmelFilterBank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=sr // 2, freeze_parameters=True)


# Function to extract features
def extract_features(file_path):
    sr, y = wav.read(file_path)
    
    # Convert audio to floating-point
    y = y.astype(np.float32)
    
    # Convert to torch tensor
    y_tensor = torch.tensor(y).unsqueeze(0)  # add batch dimension
    
    # STFT
    stft_feature = stft_transform(y_tensor).numpy()
    
    # log STFT
    log_stft_feature = log_stft_transform(y_tensor).numpy()
    
    # Spectrogram for Log Mel Filter Bank
    spectrogram = spectrogram_transform(y_tensor)
    
    # Spectrogram for Log Mel Filter Bank
    s_spectrogram = s_spectrogram_transform(y_tensor)
    
    # Log Mel Filter Bank
    logmel_feature = mel_transform(spectrogram).numpy()
    
    # Log Mel Filter Bank
    s_logmel_feature = s_mel_transform(s_spectrogram).numpy()
    
    return stft_feature.squeeze(0), log_stft_feature.squeeze(0), logmel_feature.squeeze(0), s_logmel_feature.squeeze(0)


# Iterate through dataset and save features
features = {'STFT': [], 'log_STFT': [], 'LogMel': [], 's_LogMel': []}

for category in categories:
    category_dir = os.path.join(base_dir, category)
    for recording in os.listdir(category_dir):
        recording_dir = os.path.join(category_dir, recording)
        for file in os.listdir(recording_dir):
            if file.endswith('.wav'):
                file_path = os.path.join(recording_dir, file)
                stft_feature, log_stft_feature, logmel_feature, s_logmel_feature = extract_features(file_path)
                features['STFT'].append(stft_feature)
                features['log_STFT'].append(log_stft_feature)
                features['LogMel'].append(logmel_feature)
                features['s_LogMel'].append(s_logmel_feature)
                

# Save each feature set in a separate .npz file
np.savez_compressed('tb_logs/stft_features.npz', stft_features=np.array(features['STFT']))
np.savez_compressed('tb_logs/log_stft_features.npz', log_stft_features=np.array(features['log_STFT']))
np.savez_compressed('tb_logs/logmel_features.npz', logmel_features=np.array(features['LogMel']))
np.savez_compressed('tb_logs/s_logmel_features.npz', s_logmel_features=np.array(features['s_LogMel']))


stft_shape = np.array(features['STFT']).shape 
log_stft_shape = np.array(features['log_STFT']).shape 
logmel_shape = np.array(features['LogMel']).shape
s_logmel_shape = np.array(features['s_LogMel']).shape

print(stft_shape,log_stft_shape,logmel_shape,s_logmel_shape)


