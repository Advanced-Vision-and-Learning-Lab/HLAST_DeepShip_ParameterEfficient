import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

from Demo_Parameters import Parameters
from Utils.LitModel import LitModel

# DeepShip Imports
from Datasets.Get_preprocessed_data import process_data as process_deepship_data
from Datasets.SSDataModule import SSAudioDataModule

# ShipsEar Imports
from Datasets.ShipsEar_Data_Preprocessing import Generate_Segments
from Datasets.ShipsEar_dataloader import ShipsEarDataModule

# VTUAD Imports
from Datasets.Create_Combined_VTUAD import Create_Combined_VTUAD
from Datasets.VTUAD_DataModule import AudioDataModule

from types import SimpleNamespace

def compute_layer_cosine_similarity(features_dict, model_names, num_layers=12):
    """
    Compute cosine similarity of feature maps for each sample in each batch,
    compared to the full fine-tune model, and return mean and std.
    """
    cosine_similarities = {layer: {model: [] for model in model_names if model != 'full_fine_tune'} for layer in range(num_layers)}

    reference_model = 'full_fine_tune'

    for layer in range(num_layers):
        ref_features = features_dict[reference_model][layer]
        for model_name in model_names:
            if model_name == reference_model:
                continue
            comp_features = features_dict[model_name][layer]
            # Compute cosine similarity for each sample
            for ref_sample, comp_sample in zip(ref_features, comp_features):
                ref_sample_np = ref_sample.view(1, -1).cpu().numpy()
                comp_sample_np = comp_sample.view(1, -1).cpu().numpy()
                cos_sim = cosine_similarity(ref_sample_np, comp_sample_np)[0][0]
                cosine_similarities[layer][model_name].append(cos_sim)

    # Compute mean and std for each layer and model
    cosine_sim_stats = {layer: {} for layer in range(num_layers)}
    for layer in range(num_layers):
        for model in cosine_similarities[layer]:
            similarities = cosine_similarities[layer][model]
            mean = np.mean(similarities) if similarities else 0
            std = np.std(similarities) if similarities else 0
            cosine_sim_stats[layer][model] = {'mean': mean, 'std': std}

    return cosine_sim_stats

def plot_cosine_similarity(cosine_sim_stats, model_names, output_path='cosine_similarity_plot.png'):
    """
    Plot the cosine similarity values for each layer across models (excluding full_fine_tune) and save the plot.
    """
    num_layers = len(cosine_sim_stats)
    layers = np.arange(num_layers)

    plt.figure(figsize=(12, 8))  

    # Plot each model's similarity values, skipping 'full_fine_tune'
    for model_name in model_names:
        if model_name == 'full_fine_tune':
            continue
        means = [cosine_sim_stats[layer][model_name]['mean'] for layer in layers]
        stds = [cosine_sim_stats[layer][model_name]['std'] for layer in layers]
        plt.plot(layers, means, label=model_name, marker='o', linestyle='-', linewidth=2)
        plt.fill_between(layers, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.15)

    # Set plot titles and labels
    #plt.title('Cosine Similarity for Each Layer Across Models', fontsize=20)
    plt.xlabel('Layer Number', fontsize=18)
    plt.ylabel('Cosine Similarity', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)

    plt.xticks(layers, fontsize=16)
    plt.yticks(fontsize=16)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


DATASET_CONFIG = {
    'DeepShip': {
        'process_data': process_deepship_data,
        'DataModule': SSAudioDataModule,
    },
    'ShipsEar': {
        'process_data': Generate_Segments,
        'DataModule': ShipsEarDataModule,
    },
    'VTUAD': {
        'process_data': Create_Combined_VTUAD,
        'DataModule': AudioDataModule,
    }
}

MODE_CONFIG = {
    'full_fine_tune': {
        'model': 'AST',
        'histograms_shared': True,
        'adapters_shared': True,
        'data_selection': 0,
        'numBins': 16,
        'RR': 64,
        'train_mode': 'full_fine_tune',
        'use_pretrained': True,
        'train_batch_size': 64,
        'val_batch_size': 128,
        'test_batch_size': 128,
        'num_epochs': 1,
        'num_workers': 8,
        'lr': 1e-5,
        'audio_feature': 'LogMelFBank',
        'patience': 20,
        'window_length': 2048,
        'hop_length': 512,
        'number_mels': 128,
        'sample_rate': 16000,
        'segment_length': 5,
        'adapter_location': 'None',
        'adapter_mode': 'None',
        'histogram_location': 'None',
        'histogram_mode': 'None'
    },
    'linear_probing': {
        'model': 'AST',
        'histograms_shared': True,
        'adapters_shared': True,
        'data_selection': 0,
        'numBins': 16,
        'RR': 64,
        'train_mode': 'linear_probing',
        'use_pretrained': True,
        'train_batch_size': 64,
        'val_batch_size': 128,
        'test_batch_size': 128,
        'num_epochs': 1,
        'num_workers': 8,
        'lr': 1e-3,
        'audio_feature': 'LogMelFBank',
        'patience': 20,
        'window_length': 2048,
        'hop_length': 512,
        'number_mels': 128,
        'sample_rate': 16000,
        'segment_length': 5,
        'adapter_location': 'None',
        'adapter_mode': 'None',
        'histogram_location': 'None',
        'histogram_mode': 'None'
    },
    'adapters': {
        'model': 'AST',
        'histograms_shared': True,
        'adapters_shared': True,
        'data_selection': 0,
        'numBins': 16,
        'RR': 64,
        'train_mode': 'adapters',
        'use_pretrained': True,
        'train_batch_size': 64,
        'val_batch_size': 128,
        'test_batch_size': 128,
        'num_epochs': 1,
        'num_workers': 8,
        'lr': 1e-3,
        'audio_feature': 'LogMelFBank',
        'patience': 20,
        'window_length': 2048,
        'hop_length': 512,
        'number_mels': 128,
        'sample_rate': 16000,
        'segment_length': 5,
        'adapter_location': 'mhsa',
        'adapter_mode': 'parallel',
        'histogram_location': 'None',
        'histogram_mode': 'None'
    },
    'histogram': {
        'model': 'AST',
        'histograms_shared': True,
        'adapters_shared': True,
        'data_selection': 0,
        'numBins': 16,
        'RR': 64,
        'train_mode': 'histogram',
        'use_pretrained': True,
        'train_batch_size': 64,
        'val_batch_size': 128,
        'test_batch_size': 128,
        'num_epochs': 1,
        'num_workers': 8,
        'lr': 1e-3,
        'audio_feature': 'LogMelFBank',
        'patience': 20,
        'window_length': 2048,
        'hop_length': 512,
        'number_mels': 128,
        'sample_rate': 16000,
        'segment_length': 5,
        'adapter_location': 'None',
        'adapter_mode': 'None',
        'histogram_location': 'mhsa',
        'histogram_mode': 'parallel'
    }
}

# Define TRAINING_MODES based on MODE_CONFIG keys
TRAINING_MODES = list(MODE_CONFIG.keys())

layer_outputs = {}

# Hook function to capture layer outputs
def hook_fn(layer_name, module, input, output):
    layer_outputs[layer_name] = output

# Register hooks for each layer
def register_hooks(model):
    hooks = []
    try:
        for i, blk in enumerate(model.model_ft.v.blocks):
            hook = blk.mlp.register_forward_hook(partial(hook_fn, f'block_{i}_mlp'))
            hooks.append(hook)
    except AttributeError:
        print("Error: Model architecture does not match expected structure for hook registration.")
    return hooks

# Extract features from model layers
# Setting num_batches=10 will process only the first 10 batches 
def extract_features(model, dataloader, device, num_batches=None):
    model.to(device)
    model.eval()
    features = []
    layer_outputs.clear()

    hooks = register_hooks(model)

    total_samples = 0
    total_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break

            inputs, _ = batch
            total_samples += inputs.size(0)  # Add the number of samples in this batch
            total_batches += 1  # Increment the batch count

            inputs = inputs.to(device)
            _ = model(inputs)

            batch_features = [[sample.clone() for sample in value] for value in layer_outputs.values()]
            if batch_features:
                features.extend(zip(*batch_features))

    for hook in hooks:
        hook.remove()

    return features, total_samples, total_batches

def load_model_with_args(model_path, model_name, num_classes, run_number, params):
    try:
        model = LitModel.load_from_checkpoint(
            checkpoint_path=model_path,
            Params=params,
            model_name=model_name,
            num_classes=num_classes,
            pretrained_loaded=True,
            run_number=run_number
        )
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}' from '{model_path}': {e}")
        return None

# Process each dataset
def process_dataset(dataset_name, dataset_folders, tb_logs_base_dir, features_base_dir, device, num_batches=None):
    print(f"\nProcessing Dataset: {dataset_name}")

    model_dirs = {}
    for mode in TRAINING_MODES:
        # Find folders that include the mode name
        matching_folders = [
            folder for folder in dataset_folders
            if mode.lower() in folder.lower()
        ]
        if not matching_folders:
            print(f"Warning: No folder found for mode '{mode}' in dataset '{dataset_name}'. Skipping this mode.")
            continue
        # Select the first matching folder (unique per mode)
        selected_folder = matching_folders[0]
        model_dirs[mode] = os.path.join(tb_logs_base_dir, selected_folder)

    if 'full_fine_tune' not in model_dirs:
        print(f"Error: Reference model 'full_fine_tune' not found for dataset '{dataset_name}'. Skipping.")
        return

    model_names = list(model_dirs.keys())
    
    # === Print Selected Folder Names ===
    print(f"Selected folders for dataset '{dataset_name}':")
    for mode, folder_path in model_dirs.items():
        folder_name = os.path.basename(folder_path)
        print(f"  {mode}: {folder_name}")
    # ===================================

    print(f"Found models: {model_names}")

    try:
        data_module_class = DATASET_CONFIG[dataset_name]['DataModule']
        data_module = data_module_class()
        data_module.prepare_data()        # Ensure data is prepared
        data_module.setup(stage='test')   # Existing setup call
        dataloader = data_module.test_dataloader()
    except Exception as e:
        print(f"Error initializing data module for dataset '{dataset_name}': {e}. Skipping.")
        return

    features_dict = {}

    for model_name in model_names:
        model_dir = model_dirs[model_name]
        run_folders = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('Run_')]
        if not run_folders:
            print(f"No run folders found in '{model_dir}'. Skipping model '{model_name}'.")
            continue
        selected_run = run_folders[0]
        checkpoint_dir = os.path.join(model_dir, selected_run, 'metrics', 'version_0', 'checkpoints')
        if not os.path.isdir(checkpoint_dir):
            print(f"Checkpoints directory not found in '{checkpoint_dir}'. Skipping model '{model_name}'.")
            continue
        ckpt_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
        if not ckpt_files:
            print(f"No checkpoint found in '{checkpoint_dir}'. Skipping model '{model_name}'.")
            continue
        model_path = ckpt_files[0]
        print(f"Loading model '{model_name}' from '{model_path}'...")

        # Retrieve mode-specific parameters
        mode_config = MODE_CONFIG.get(model_name, None)
        if mode_config is None:
            print(f"No configuration found for training mode '{model_name}'. Skipping.")
            continue

        # === Dataset-Specific Parameter Adjustment ===
        if dataset_name == 'VTUAD':
            mode_config = mode_config.copy()  # Create a copy to avoid mutating the original config
            mode_config['sample_rate'] = 32000
            mode_config['segment_length'] = 1
            print(f"Adjusted parameters for VTUAD: sample_rate={mode_config['sample_rate']}, segment_length={mode_config['segment_length']}")
        # ============================================

        # === Set Number of Classes Based on Dataset ===
        if dataset_name == 'DeepShip':
            num_classes = 4
        else:
            num_classes = 5
        # =============================================

        try:
            # Create an args-like object using SimpleNamespace
            args = SimpleNamespace(**mode_config)
            # Instantiate Parameters with mode-specific configuration
            params = Parameters(args)
            
            # SET CORRECT NUMBER OF CLASSES HERE 
            model = load_model_with_args(model_path, model_name, num_classes, 0, params)
            if not model:
                continue

            features, total_samples, total_batches = extract_features(
                model, dataloader, device, num_batches=num_batches
            )
            if features:
                transposed_features = list(zip(*features))
                features_dict[model_name] = transposed_features

            # Print the number of samples and batches used
            print(f"Model '{model_name}': {total_samples} samples from {total_batches} batches used for similarity analysis.")
        except Exception as e:
            print(f"Error processing model '{model_name}': {e}. Skipping.")

    # After processing all models, compute cosine similarity 
    if 'full_fine_tune' in features_dict:
        cosine_sim_stats = compute_layer_cosine_similarity(features_dict, model_names)
        output_plot_path = os.path.join(features_base_dir, f"{dataset_name}_cosine_similarity_plot.png")
        plot_cosine_similarity(cosine_sim_stats, model_names, output_path=output_plot_path)
        print(f"Cosine similarity plot saved to '{output_plot_path}'.")
    else:
        print(f"Reference model 'full_fine_tune' features not available for dataset '{dataset_name}'. Skipping cosine similarity computation.")

def traverse_tb_logs(tb_logs_base_dir):
    dataset_folders = {}
    for entry in os.listdir(tb_logs_base_dir):
        entry_path = os.path.join(tb_logs_base_dir, entry)
        if os.path.isdir(entry_path):
            dataset_name = entry.split('_')[0]
            if dataset_name not in DATASET_CONFIG:
                print(f"Warning: Dataset '{dataset_name}' is not recognized. Skipping folder '{entry}'.")
                continue
            if dataset_name not in dataset_folders:
                dataset_folders[dataset_name] = []
            dataset_folders[dataset_name].append(entry)
    return dataset_folders

def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments with mode-specific parameters')
    return parser.parse_args()

def main():

    tb_logs_base_dir = 'tb_logs'
    features_base_dir = os.path.join('features', 'similarity_plots')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_batches = None

    if not os.path.isdir(tb_logs_base_dir):
        print(f"Error: tb_logs directory '{tb_logs_base_dir}' does not exist.")
        return

    os.makedirs(features_base_dir, exist_ok=True)

    dataset_folders = traverse_tb_logs(tb_logs_base_dir)
    if not dataset_folders:
        print("No valid datasets found in tb_logs. Exiting.")
        return

    max_workers = min(len(dataset_folders), os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for dataset_name, folders in dataset_folders.items():
            future = executor.submit(
                process_dataset,
                dataset_name,
                folders,
                tb_logs_base_dir,
                features_base_dir,
                device,
                num_batches
            )
            futures.append(future)

        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during processing: {e}")

    print("\nAll datasets processed.")

if __name__ == "__main__":
    main()
