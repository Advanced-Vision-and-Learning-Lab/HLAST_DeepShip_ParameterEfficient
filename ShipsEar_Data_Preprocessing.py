import os
import librosa
import soundfile as sf
import math
import pandas as pd

def Generate_Segments(dataset_dir, target_sr=16000, segment_length=5):
    '''
    dataset_dir: Directory containing ShipsEar data folder and csv
    target_sr: Desired sampling rate in Hz
    segment_length: Desired segment length in seconds
    '''
    data_csv = pd.read_excel('{}{}'.format(dataset_dir, 'shipsEar.xlsx'))
    
    # Define groups based on ShipsEar paper
    A = ['Fishboat', 'Trawler', 'Mussel boat', 'Tugboat', 'Dredger']
    B = ['Motorboat', 'Pilot ship', 'Sailboat']
    C = ['Passengers']
    D = ['Ocean liner', 'RORO']
    E = ['Natural ambient noise']
    
    # Set the folder path containing the WAV files and subfolders
    ship_type = ['A', 'B', 'C', 'D', 'E']
    ships_dictionary = {'A': A, 'B': B, 'C': C, 'D': D, 'E': E}

    # Flag to track if any segmentation is needed
    segmentation_needed = False
    
    # First pass to check if all files are already segmented
    for ship in ship_type:
        segments_path = '{}/{}'.format(dataset_dir, ship)
        temp_vessels = ships_dictionary[ship]
        temp_samples = data_csv[data_csv['Type'].isin(temp_vessels)]
       
        for index, row in temp_samples.iterrows():
            subfolder_name = row['Filename'].split('.')[0]
            segment_folder_path = os.path.join(segments_path, subfolder_name)
            
            # Check if any file is not segmented yet
            if not os.path.exists(segment_folder_path):
                segmentation_needed = True
                break  # No need to check further if we know segmentation is needed

        if segmentation_needed:
            break  # Exit outer loop as well

    # Print whether segmentation will be performed or skipped based on the check
    if segmentation_needed:
        print("Segmentation will be performed.")
    else:
        print("All files are already segmented. Skipping segmentation.")
        return  # Exit early since no segmentation is needed

    # Proceed with segmentation if needed
    for ship in ship_type:
        segments_path = '{}/{}'.format(dataset_dir, ship)
        
        temp_vessels = ships_dictionary[ship]
        temp_samples = data_csv[data_csv['Type'].isin(temp_vessels)]
       
        for index, row in temp_samples.iterrows():
            subfolder_name = row['Filename'].split('.')[0]
            file_name = row['Filename']
            
            segment_folder_path = os.path.join(segments_path, subfolder_name)
            if os.path.exists(segment_folder_path):
                continue  # Skip this file since segments already exist

            # Load the audio signal
            file_path = os.path.join(dataset_dir, file_name)
            audio, sr = librosa.load(file_path, sr=None)

            # Resample to the target sampling rate
            audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

            # Divide the resampled audio into segments and save them to a folder
            duration = len(audio_resampled)
            segment_duration = target_sr * segment_length
            number_of_segments = math.ceil(duration / segment_duration)

            for i in range(number_of_segments):
                start_i = i * segment_duration
                end_i = start_i + segment_duration

                if end_i > duration:
                    end_i = duration

                output_music = audio_resampled[start_i:end_i]

                # Only save full-length segments (ignore shorter last segments)
                if end_i - start_i == segment_duration:
                    segment_file_path = os.path.join(segment_folder_path,
                                                     f'{os.path.splitext(file_name)[0]}-Segment_{i+1}.wav')
                    os.makedirs(os.path.dirname(segment_file_path), exist_ok=True)
                    sf.write(segment_file_path, output_music, samplerate=target_sr)

if __name__ == '__main__':
    
    dataset_dir = './ShipsEar/'
    Generate_Segments(dataset_dir, target_sr=16000, segment_length=5)