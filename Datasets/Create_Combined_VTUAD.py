import os
import shutil

class Create_Combined_VTUAD:
    def __init__(self, base_dir, scenarios, combined_scenario):
        """
        Initialize with base directory, list of scenarios to combine, and the name of the combined scenario.
        """
        self.base_dir = base_dir
        self.scenarios = scenarios
        self.combined_scenario = combined_scenario
        self.subfolders = ['train', 'test', 'validation']
        self.categories = ['background', 'cargo', 'passengership', 'tanker', 'tug']
    
    def create_combined_structure(self):
        """Create the directory structure for the combined scenario."""
        for subfolder in self.subfolders:
            for category in self.categories:
                path = os.path.join(self.base_dir, self.combined_scenario, subfolder, 'audio', category)
                os.makedirs(path, exist_ok=True)
    
    def copy_files_to_combined(self):
        """Copy .wav files from each scenario to the combined scenario."""
        for scenario in self.scenarios:
            # Account for the nested scenario folder
            nested_scenario_path = os.path.join(self.base_dir, scenario, scenario)
            
            for subfolder in self.subfolders:
                for category in self.categories:
                    source_folder = os.path.join(nested_scenario_path, subfolder, 'audio', category)
                    target_folder = os.path.join(self.base_dir, self.combined_scenario, subfolder, 'audio', category)
                    
                    if not os.path.exists(source_folder):
                        print(f"Source folder {source_folder} does not exist. Skipping.")
                        continue
                    
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
    
    def create_combined_scenario(self):
        """Main method to create the combined scenario."""
        self.create_combined_structure()
        self.copy_files_to_combined()
