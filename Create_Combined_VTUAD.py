import os
import shutil

# Paths to the original scenario folders
scenarios = [
    'VTUAD/inclusion_2000_exclusion_4000',
    'VTUAD/inclusion_3000_exclusion_5000',
    'VTUAD/inclusion_4000_exclusion_6000'
]

# Path to the new combined scenario folder
combined_scenario = 'VTUAD/combined_scenario'

# Folder structure inside each scenario
subfolders = ['train', 'test', 'validation']
categories = ['background', 'cargo', 'passengership', 'tanker', 'tug']

def create_combined_structure(combined_scenario):
    """Create the directory structure for the combined scenario."""
    for subfolder in subfolders:
        for category in categories:
            path = os.path.join(combined_scenario, subfolder, 'audio', category)
            os.makedirs(path, exist_ok=True)

def copy_files_to_combined(scenarios, combined_scenario):
    """Copy .wav files from each scenario to the combined scenario."""
    for scenario in scenarios:
        for subfolder in subfolders:
            for category in categories:
                source_folder = os.path.join(scenario, subfolder, 'audio', category)
                target_folder = os.path.join(combined_scenario, subfolder, 'audio', category)
                
                # Copy all .wav files from source to target
                for file_name in os.listdir(source_folder):
                    if file_name.endswith('.wav'):
                        source_file = os.path.join(source_folder, file_name)
                        target_file = os.path.join(target_folder, file_name)
                        
                        # If a file with the same name exists, rename it to avoid overwriting
                        if os.path.exists(target_file):
                            base_name, ext = os.path.splitext(file_name)
                            counter = 1
                            while os.path.exists(target_file):
                                new_file_name = f"{base_name}_{counter}{ext}"
                                target_file = os.path.join(target_folder, new_file_name)
                                counter += 1
                        
                        shutil.copy2(source_file, target_file)

if __name__ == '__main__':
    # Step 1: Create the combined folder structure
    create_combined_structure(combined_scenario)
    
    # Step 2: Copy files from each scenario into the combined folder
    copy_files_to_combined(scenarios, combined_scenario)

    print("Files have been successfully combined!")