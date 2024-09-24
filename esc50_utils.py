import os
import pandas as pd
import torchaudio
import requests
import zipfile
from tqdm import tqdm
import shutil
import hashlib
from multiprocessing import Pool
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
def prepare_esc50_dataset(original_dir, resampled_dir, target_sample_rate=16000, num_workers=8):
    # Download and prepare original dataset if not exists
    if not os.path.exists(os.path.join(original_dir, 'ESC-50-master')):
        os.makedirs(original_dir, exist_ok=True)
        url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
        expected_checksum = "e0754beec6a68076727c4bf90749653a"  # MD5 checksum of the zip file
        
        logging.info("Downloading ESC-50 dataset...")
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
        
        # Verify checksum
        with open(os.path.join(original_dir, "esc50.zip"), "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        if file_hash != expected_checksum:
            logging.warning("Downloaded file checksum mismatch. The dataset might be corrupted or updated.")
        
        logging.info("Extracting dataset...")
        with zipfile.ZipFile(os.path.join(original_dir, "esc50.zip"), 'r') as zip_ref:
            zip_ref.extractall(original_dir)
        
        os.remove(os.path.join(original_dir, "esc50.zip"))
        logging.info("ESC-50 dataset downloaded and prepared.")
    else:
        logging.info("ESC-50 dataset already downloaded and prepared.")

    # Check if resampled dataset exists and has the metadata file
    if os.path.exists(resampled_dir) and os.path.exists(os.path.join(resampled_dir, 'meta', 'esc50.csv')):
        logging.info("Resampled dataset found.")
        return

    logging.info("Resampled dataset not found or incomplete. Creating...")
    # Remove the incomplete resampled dataset if it exists
    if os.path.exists(resampled_dir):
        shutil.rmtree(resampled_dir)

    os.makedirs(resampled_dir, exist_ok=True)
    os.makedirs(os.path.join(resampled_dir, 'audio'), exist_ok=True)
    os.makedirs(os.path.join(resampled_dir, 'meta'), exist_ok=True)

    metadata = pd.read_csv(os.path.join(original_dir, 'ESC-50-master', 'meta', 'esc50.csv'))

    def resample_and_validate_file(args):
        original_path, target_path = args
        try:
            waveform, sample_rate = torchaudio.load(original_path)
            if sample_rate != target_sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
                waveform = resampler(waveform)
            torchaudio.save(target_path, waveform, target_sample_rate)
            return validate_audio(target_path, target_sample_rate)
        except Exception as e:
            logging.error(f"Error processing {original_path}: {str(e)}")
            return False

    logging.info(f"Resampling and validating audio files using {num_workers} workers...")
    with Pool(num_workers) as p:
        results = list(tqdm(
            p.imap(resample_and_validate_file, 
                    [(os.path.join(original_dir, 'ESC-50-master', 'audio', row['filename']),
                      os.path.join(resampled_dir, 'audio', row['filename']))
                    for _, row in metadata.iterrows()]),
            total=len(metadata),
            desc="Processing files"
        ))

    if not all(results):
        logging.warning("Some audio files failed validation. The dataset might be incomplete or corrupted.")

    # Copy metadata file
    metadata.to_csv(os.path.join(resampled_dir, 'meta', 'esc50.csv'), index=False)

    logging.info(f"Dataset resampled to {target_sample_rate} Hz and saved in {resampled_dir}")

def validate_audio(file_path, expected_sample_rate):
    waveform, sample_rate = torchaudio.load(file_path)
    duration_samples = 5 * expected_sample_rate  # 5 seconds
    
    if sample_rate != expected_sample_rate:
        logging.warning(f"{file_path}: Sample rate mismatch. Expected {expected_sample_rate}, got {sample_rate}")
        return False
    if waveform.shape[0] != 1:
        logging.warning(f"{file_path}: Not mono. Got {waveform.shape[0]} channels")
        return False
    if waveform.shape[1] != duration_samples:
        logging.warning(f"{file_path}: Incorrect duration. Expected {duration_samples} samples, got {waveform.shape[1]}")
        return False
    
    return True


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def prepare_esc50_dataset(original_dir, resampled_dir, target_sample_rate=16000, num_workers=8, resample=True):
#     # Download and prepare original dataset if not exists
#     if not os.path.exists(os.path.join(original_dir, 'ESC-50-master')):
#         os.makedirs(original_dir, exist_ok=True)
#         url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
#         expected_checksum = "e0754beec6a68076727c4bf90749653a"  # MD5 checksum of the zip file
        
#         logging.info("Downloading ESC-50 dataset...")
#         response = requests.get(url, stream=True)
#         total_size = int(response.headers.get('content-length', 0))
        
#         with open(os.path.join(original_dir, "esc50.zip"), "wb") as file, tqdm(
#             desc="Downloading ESC-50",
#             total=total_size,
#             unit="iB",
#             unit_scale=True,
#             unit_divisor=1024,
#         ) as progress_bar:
#             for data in response.iter_content(chunk_size=1024):
#                 size = file.write(data)
#                 progress_bar.update(size)
        
#         # Verify checksum
#         with open(os.path.join(original_dir, "esc50.zip"), "rb") as f:
#             file_hash = hashlib.md5(f.read()).hexdigest()
#         if file_hash != expected_checksum:
#             logging.warning("Downloaded file checksum mismatch. The dataset might be corrupted or updated.")
        
#         logging.info("Extracting dataset...")
#         with zipfile.ZipFile(os.path.join(original_dir, "esc50.zip"), 'r') as zip_ref:
#             zip_ref.extractall(original_dir)
        
#         os.remove(os.path.join(original_dir, "esc50.zip"))
#         logging.info("ESC-50 dataset downloaded and prepared.")
#     else:
#         logging.info("ESC-50 dataset already downloaded and prepared.")

#     if not resample:
#         logging.info("Using original dataset without resampling.")
#         return

#     # Check if resampled dataset exists and has the metadata file
#     if os.path.exists(resampled_dir) and os.path.exists(os.path.join(resampled_dir, 'meta', 'esc50.csv')):
#         logging.info("Resampled dataset found.")
#         return

#     logging.info("Resampled dataset not found or incomplete. Creating...")
#     # Remove the incomplete resampled dataset if it exists
#     if os.path.exists(resampled_dir):
#         shutil.rmtree(resampled_dir)

#     os.makedirs(resampled_dir, exist_ok=True)
#     os.makedirs(os.path.join(resampled_dir, 'audio'), exist_ok=True)
#     os.makedirs(os.path.join(resampled_dir, 'meta'), exist_ok=True)

#     metadata = pd.read_csv(os.path.join(original_dir, 'ESC-50-master', 'meta', 'esc50.csv'))

#     def resample_and_validate_file(args):
#         original_path, target_path = args
#         try:
#             waveform, sample_rate = torchaudio.load(original_path)
#             if sample_rate != target_sample_rate:
#                 resampler = torchaudio.transforms.Resample(sample_rate, target_sample_rate)
#                 waveform = resampler(waveform)
#             torchaudio.save(target_path, waveform, target_sample_rate)
#             return validate_audio(target_path, target_sample_rate)
#         except Exception as e:
#             logging.error(f"Error processing {original_path}: {str(e)}")
#             return False

#     logging.info(f"Resampling and validating audio files using {num_workers} workers...")
#     with Pool(num_workers) as p:
#         results = list(tqdm(
#             p.imap(resample_and_validate_file, 
#                    [(os.path.join(original_dir, 'ESC-50-master', 'audio', row['filename']),
#                      os.path.join(resampled_dir, 'audio', row['filename']))
#                     for _, row in metadata.iterrows()]),
#             total=len(metadata),
#             desc="Processing files"
#         ))

#     if not all(results):
#         logging.warning("Some audio files failed validation. The dataset might be incomplete or corrupted.")

#     # Copy metadata file
#     metadata.to_csv(os.path.join(resampled_dir, 'meta', 'esc50.csv'), index=False)

#     logging.info(f"Dataset resampled to {target_sample_rate} Hz and saved in {resampled_dir}")

# def validate_audio(file_path, expected_sample_rate):
#     waveform, sample_rate = torchaudio.load(file_path)
#     duration_samples = 5 * expected_sample_rate  # 5 seconds
    
#     if sample_rate != expected_sample_rate:
#         logging.warning(f"{file_path}: Sample rate mismatch. Expected {expected_sample_rate}, got {sample_rate}")
#         return False
#     if waveform.shape[0] != 1:
#         logging.warning(f"{file_path}: Not mono. Got {waveform.shape[0]} channels")
#         return False
#     if waveform.shape[1] != duration_samples:
#         logging.warning(f"{file_path}: Incorrect duration. Expected {duration_samples} samples, got {waveform.shape[1]}")
#         return False
    
#     return True













