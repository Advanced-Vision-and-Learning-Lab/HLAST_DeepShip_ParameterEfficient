import os
import pandas as pd
import torchaudio
import requests
import zipfile
from tqdm import tqdm

import shutil


def prepare_esc50_dataset(original_dir, resampled_dir, target_sample_rate=16000):
    # Download and prepare original dataset if not exists
    if not os.path.exists(os.path.join(original_dir, 'ESC-50-master')):
        os.makedirs(original_dir, exist_ok=True)
        url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(os.path.join(original_dir, "esc50.zip"), "wb") as file, tqdm(
            desc="Downloading ESC-50",
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                progress_bar.update(size)
        
        with zipfile.ZipFile(os.path.join(original_dir, "esc50.zip"), 'r') as zip_ref:
            zip_ref.extractall(original_dir)
        
        os.remove(os.path.join(original_dir, "esc50.zip"))
        print("ESC-50 dataset downloaded and prepared.")
    else:
        print("ESC-50 dataset already downloaded and prepared.")

    # Check if resampled dataset exists and has the metadata file
    if os.path.exists(resampled_dir) and os.path.exists(os.path.join(resampled_dir, 'meta', 'esc50.csv')):
        print("Resampled dataset found.")
        return

    print("Resampled dataset not found or incomplete. Creating...")
    # Remove the incomplete resampled dataset if it exists
    if os.path.exists(resampled_dir):
        shutil.rmtree(resampled_dir)

    os.makedirs(resampled_dir, exist_ok=True)
    os.makedirs(os.path.join(resampled_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(resampled_dir, 'meta'), exist_ok=True)

    metadata = pd.read_csv(os.path.join(original_dir, 'ESC-50-master', 'meta', 'esc50.csv'))

    for _, row in tqdm(metadata.iterrows(), total=len(metadata), desc="Resampling audio files"):
        original_path = os.path.join(original_dir, 'ESC-50-master', 'audio', row['filename'])
        target_path = os.path.join(resampled_dir, 'audio', row['filename'])

        waveform, sample_rate = torchaudio.load(original_path)
        
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
            waveform = resampler(waveform)
        
        torchaudio.save(target_path, waveform, target_sample_rate)

    # Copy metadata file
    metadata.to_csv(os.path.join(resampled_dir, 'meta', 'esc50.csv'), index=False)

    print(f"Dataset resampled to {target_sample_rate} Hz and saved in {resampled_dir}")
