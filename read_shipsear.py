import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Function to read and display information about a .wav file
def read_wav_info(wav_path):
    # Read the wav file using scipy.io.wavfile.read
    sample_rate, data = wavfile.read(wav_path)
    
    # Print basic information about the file
    print(f"Sample Rate: {sample_rate} Hz")
    print(f"Data Type: {data.dtype}")
    
    # Check if the audio is mono or stereo
    if len(data.shape) == 1:
        print("Number of Channels: 1 (Mono)")
    else:
        print(f"Number of Channels: {data.shape[1]} (Stereo)")
    
    # Calculate the duration of the audio in seconds
    duration = data.shape[0] / sample_rate
    print(f"Duration: {duration:.2f} seconds")
    
    # Plot the waveform for the first few seconds (up to 5 seconds)
    time = np.linspace(0., duration, data.shape[0])
    
    plt.figure(figsize=(10, 4))
    
    if len(data.shape) == 1:
        # Mono audio
        plt.plot(time, data, label="Mono Channel")
    else:
        # Stereo audio, plot both channels
        plt.plot(time, data[:, 0], label="Left Channel")
        plt.plot(time, data[:, 1], label="Right Channel")
    
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Waveform of the Audio File")
    plt.show()

# Example usage with a sample .wav file path
wav_file_path = 'shipsEar/Fishboat/73__23_07_13_H3_pesqMariCarmen.wav'  # Replace with your actual file path
read_wav_info(wav_file_path)