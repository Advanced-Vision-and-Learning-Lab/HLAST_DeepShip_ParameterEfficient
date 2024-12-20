import os
import librosa
import soundfile as sf
import math
from multiprocessing import Pool

def process_file(args):
    file_path, segments_path, ship, target_sr, segment_length = args
    audio, sr = librosa.load(file_path, sr=None)
    audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    duration = len(audio_resampled)
    segment_duration = target_sr * segment_length
    number = math.ceil(duration / segment_duration)

    for i in range(number):
        start_i = int(i * segment_duration)
        end_i = int(start_i + segment_duration)
        if end_i > duration:
            end_i = duration
        output_music = audio_resampled[start_i:end_i]
        if end_i - start_i == segment_duration:
            subfolder_name = os.path.basename(os.path.dirname(file_path))
            file_name = os.path.basename(file_path)
            segment_file_path = os.path.join(segments_path, subfolder_name,
                                             f'{os.path.splitext(file_name)[0]}_{ship}-Segment_{i + 1}.wav')
            os.makedirs(os.path.dirname(segment_file_path), exist_ok=True)
            sf.write(segment_file_path, output_music, samplerate=target_sr)

def Generate_Segments(dataset_dir, segments_dir, target_sr=16000, segment_length=3, num_workers=8):
    ship_types = ['Cargo', 'Passengership', 'Tanker', 'Tug']
    
    tasks = []
    for ship in ship_types:
        folder_path = f'{dataset_dir}{ship}'
        segments_path = f'{segments_dir}{ship}'

        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)
            if not os.path.isdir(subfolder_path):
                continue

            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.wav'):
                    file_path = os.path.join(subfolder_path, file_name)
                    tasks.append((file_path, segments_path, ship, target_sr, segment_length))

    with Pool(processes=num_workers) as pool:
        pool.map(process_file, tasks)

def process_data(data_dir='./Datasets/DeepShip/', sample_rate=None, segment_length=None, num_workers=8):
    segments_dir = f'{data_dir}Segments_{segment_length}s_{sample_rate}hz/'

    if not os.path.exists(segments_dir):
        os.makedirs(segments_dir)
        print(f"Segments folder is creating at {segments_dir}")
        Generate_Segments(data_dir, segments_dir, target_sr=sample_rate, segment_length=segment_length, num_workers=num_workers)
    else:
        print("Segments folder already exists. Skipping segment generation.")

if __name__ == "__main__":
    process_data(num_workers=4) 
