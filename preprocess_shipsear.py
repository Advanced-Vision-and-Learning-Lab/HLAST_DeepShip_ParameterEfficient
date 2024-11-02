import os
from scipy.io import wavfile
from scipy.signal import resample
import numpy as np

# Function to process and segment a .wav file
def process_and_segment_wav(input_path, output_dir, folder_name, target_sample_rate=16000, segment_duration=5):
    """
    Process and segment a .wav file by resampling it to 16000 Hz and splitting it into 5-second segments.
    
    Args:
        input_path (str): Path to the input .wav file.
        output_dir (str): Path to the directory where processed files will be saved.
        folder_name (str): Name of the folder (class label) in which to save the processed files.
        target_sample_rate (int): The desired sample rate for resampling.
        segment_duration (int): The duration of each audio segment in seconds.
    
    Returns:
        None
    """
    # Read the wav file
    sample_rate, data = wavfile.read(input_path)
    
    # Calculate the number of samples per segment (5 seconds)
    segment_samples = segment_duration * target_sample_rate
    
    # Resample the audio if necessary (from original sample rate to 16000 Hz)
    if sample_rate != target_sample_rate:
        num_samples = int(len(data) * target_sample_rate / sample_rate)
        data = resample(data, num_samples)
    
    # Ensure data is in int16 format (standard for audio)
    data = np.int16(data / np.max(np.abs(data)) * 32767)

    # Split the data into 5-second segments
    total_samples = len(data)
    num_segments = total_samples // segment_samples
    
    # Ensure output directory exists for this class
    output_class_dir = os.path.join(output_dir, folder_name)
    if not os.path.exists(output_class_dir):
        os.makedirs(output_class_dir)

    for i in range(num_segments):
        start_idx = i * segment_samples
        end_idx = start_idx + segment_samples
        
        # Extract the segment
        segment_data = data[start_idx:end_idx]
        
        # Create output filename for each segment
        output_filename = f"{os.path.splitext(os.path.basename(input_path))[0]}_segment_{i}.wav"
        
        # Save the processed segment
        output_path = os.path.join(output_class_dir, output_filename)
        wavfile.write(output_path, target_sample_rate, segment_data)

# Main function to process all wav files in a directory while maintaining folder structure
def preprocess_dataset(input_dir, output_dir):
    """
    Preprocess all .wav files in a dataset by resampling and segmenting them. The function skips processing if 
    the output directory already exists.
    
    Args:
        input_dir (str): Path to the root directory containing original .wav files organized by class folders.
        output_dir (str): Path to the directory where processed files will be saved.
    
    Returns:
        None
    """
    # Check if the output directory exists. If it does, skip processing.
    if os.path.exists(output_dir):
        print(f"Output directory exists: {output_dir}. Skipping processing.")
        return  # Skip further processing since the output directory already exists

    print(f"Output directory does not exist. Creating and processing: {output_dir}")
    
    # Create output directory since it doesn't exist yet
    os.makedirs(output_dir)

    # Process all .wav files in input directory
    for folder_name in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder_name)
        
        if os.path.isdir(folder_path):
            for wav_file in os.listdir(folder_path):
                if wav_file.endswith(".wav"):
                    input_wav_path = os.path.join(folder_path, wav_file)
                    process_and_segment_wav(input_wav_path, output_dir, folder_name)

# Example usage:
# preprocess_if_needed('shipsEar', 'processed_shipsEar')