import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from functools import partial
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
from types import SimpleNamespace
import h5py

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
        #'numBins': 16, #DeepShip
        #'RR': 128, #DeepShip
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
        #'numBins': 16, #DeepShip
        #'RR': 128, #DeepShip
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
        'histograms_shared': False,
        'adapters_shared': False,
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
        'histograms_shared': False,
        'adapters_shared': False,
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

TRAINING_MODES = list(MODE_CONFIG.keys())

layer_outputs = {}

def hook_fn(layer_name, module, input, output):
    layer_outputs[layer_name] = output

def register_hooks(model):
    hooks = []
    try:
        for i, blk in enumerate(model.model_ft.v.blocks):
            hook = blk.mlp.register_forward_hook(partial(hook_fn, f'block_{i}_mlp'))
            hooks.append(hook)
    except AttributeError:
        print("Error: Model architecture does not match expected structure for hook registration.")
    return hooks


def extract_features(model, dataloader, device, h5_file_path, num_batches=None):
    """
    Extract features and store them incrementally in an HDF5 file to save memory.
    """
    model.to(device)
    model.eval()
    layer_outputs.clear()
    hooks = register_hooks(model)

    if os.path.exists(h5_file_path):
        os.remove(h5_file_path)

    h5_file = h5py.File(h5_file_path, 'w')
    dataset = None
    total_samples = 0
    total_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break

            inputs, _ = batch
            batch_size = inputs.size(0)
            total_samples += batch_size
            total_batches += 1

            inputs = inputs.to(device)
            _ = model(inputs)

            # layer_outputs: dict of {layer_name: output}
            # Convert to a consistent format: [num_layers, batch_size, feature_dim]
            # Currently: batch_features is [[sample for each sample in batch] for each layer]
            batch_features = [[sample.clone() for sample in value] for value in layer_outputs.values()]

            # We have: len(batch_features) = num_layers
            # Each batch_features[i] is a list of length batch_size, each a tensor
            # Flatten each sample and stack
            num_layers = len(batch_features)
            # Assume all samples have the same shape
            with torch.no_grad():
                sample_flat = batch_features[0][0].view(1, -1)  # Flatten example
                feature_dim = sample_flat.size(-1)

            # Create a NumPy array for the entire batch
            # Shape: (batch_size, num_layers, feature_dim)
            batch_array = np.zeros((batch_size, num_layers, feature_dim), dtype=np.float32)

            for layer_idx, layer_data in enumerate(batch_features):
                for sample_idx, sample_tensor in enumerate(layer_data):
                    # Flatten and move to CPU numpy
                    sample_np = sample_tensor.view(1, -1).cpu().numpy().astype(np.float32)
                    batch_array[sample_idx, layer_idx, :] = sample_np

            # Initialize the dataset if this is the first batch
            if dataset is None:
                maxshape = (None, num_layers, feature_dim)
                dataset = h5_file.create_dataset('features', data=batch_array,
                                                 maxshape=maxshape, chunks=(64, num_layers, feature_dim))
            else:
                # Append new batch
                old_size = dataset.shape[0]
                new_size = old_size + batch_size
                dataset.resize(new_size, axis=0)
                dataset[old_size:new_size, :, :] = batch_array

    for hook in hooks:
        hook.remove()

    h5_file.close()
    return total_samples, total_batches

def compute_layer_cosine_similarity(h5_files_dict, model_names, output_path='cosine_similarity_plot.png'):
    """
    Compute cosine similarity from h5 files (stored for each model).
    We assume each file has a dataset 'features' of shape (num_samples, num_layers, feature_dim).
    We only load in chunks if needed to prevent memory issues.
    """
    reference_model = 'full_fine_tune'
    if reference_model not in h5_files_dict:
        print("Reference model (full_fine_tune) not found. Cannot compute similarity.")
        return

    # Open reference model file to determine dimensions
    with h5py.File(h5_files_dict[reference_model], 'r') as f_ref:
        ref_data = f_ref['features']
        num_samples, num_layers, feature_dim = ref_data.shape

    # We'll compute similarities layer by layer
    cosine_similarities = {layer: {m: [] for m in model_names if m != reference_model} for layer in range(num_layers)}

    # Load reference model data fully into memory 
    with h5py.File(h5_files_dict[reference_model], 'r') as f_ref:
        ref_data = f_ref['features'][:]  # shape (num_samples, num_layers, feature_dim)

    for model_name in model_names:
        if model_name == reference_model:
            continue
        with h5py.File(h5_files_dict[model_name], 'r') as f_comp:
            comp_data = f_comp['features'][:]  # shape (num_samples, num_layers, feature_dim)

        # Compute similarity per layer
        # We'll do vectorized cosine similarity if possible
        # Cosine similarity: (A·B) / (||A||*||B||)
        # Here: A and B are (num_samples, feature_dim)
        for layer in range(num_layers):
            A = ref_data[:, layer, :]  # (num_samples, feature_dim)
            B = comp_data[:, layer, :] # (num_samples, feature_dim)

            # Normalize
            A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
            B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)

            # Compute element-wise cos sim
            cos_sims = np.sum(A_norm * B_norm, axis=1)  # (num_samples,)
            cosine_similarities[layer][model_name] = cos_sims.tolist()

    # Compute mean/std
    cosine_sim_stats = {layer: {} for layer in range(num_layers)}
    for layer in range(num_layers):
        for model in cosine_similarities[layer]:
            sims = cosine_similarities[layer][model]
            mean = np.mean(sims) if sims else 0
            std = np.std(sims) if sims else 0
            cosine_sim_stats[layer][model] = {'mean': mean, 'std': std}

    # Plot the results
    plot_cosine_similarity(cosine_sim_stats, model_names, output_path=output_path)

def plot_cosine_similarity(cosine_sim_stats, model_names, output_path='cosine_similarity_plot.png'):
    num_layers = len(cosine_sim_stats)
    layers = np.arange(num_layers)

    plt.figure(figsize=(10, 6))  
    for model_name in model_names:
        if model_name == 'full_fine_tune':
            continue
        means = [cosine_sim_stats[layer][model_name]['mean'] for layer in layers]
        stds = [cosine_sim_stats[layer][model_name]['std'] for layer in layers]
        plt.plot(layers, means, label=model_name, marker='o', linestyle='-', linewidth=2)
        plt.fill_between(layers, np.array(means) - np.array(stds), np.array(means) + np.array(stds), alpha=0.10)

    plt.xlabel('Layer Number', fontsize=18)
    plt.ylabel('Cosine Similarity', fontsize=18)
    plt.legend(fontsize=16, loc='lower left')
    plt.grid(True)
    plt.xticks(layers, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylim(0.15, 1.00)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

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

def process_dataset(dataset_name, dataset_folders, tb_logs_base_dir, features_base_dir, device, num_batches=None):
    print(f"\nProcessing Dataset: {dataset_name}")

    model_dirs = {}
    for mode in TRAINING_MODES:
        mode_config = MODE_CONFIG.get(mode, {}).copy()
        search_substrings = []
        
        if mode_config.get('adapters_shared') is not None:
            search_substrings.append(f"AdaptShared{mode_config['adapters_shared']}")
        if mode_config.get('histograms_shared') is not None:
            search_substrings.append(f"Shared{mode_config['histograms_shared']}")
        if mode_config.get('RR') is not None:
            search_substrings.append(f"RR{mode_config['RR']}")
        if mode_config.get('numBins') is not None:
            search_substrings.append(f"{mode_config['numBins']}bins")
        if mode_config.get('adapter_location') != 'None':
            search_substrings.append(mode_config['adapter_location'])
        if mode_config.get('adapter_mode') != 'None':
            search_substrings.append(mode_config['adapter_mode'])
        if mode_config.get('histogram_location') != 'None':
            search_substrings.append(mode_config['histogram_location'])
        if mode_config.get('histogram_mode') != 'None':
            search_substrings.append(mode_config['histogram_mode'])

        matching_folders = [
            folder for folder in dataset_folders
            if mode.lower() in folder.lower() and all(sub in folder for sub in search_substrings)
        ]

        if not matching_folders:
            print(f"Warning: No folder found for mode '{mode}' in dataset '{dataset_name}'. Skipping this mode.")
            continue

        if dataset_name == 'VTUAD':
            mode_config['sample_rate'] = 32000
            mode_config['segment_length'] = 1
            print(f"Adjusted parameters for VTUAD: sample_rate={mode_config['sample_rate']}, segment_length={mode_config['segment_length']}")

        selected_folder = matching_folders[0]
        model_dirs[mode] = {
            'folder_path': os.path.join(tb_logs_base_dir, selected_folder),
            'mode_config': mode_config
        }

    if 'full_fine_tune' not in model_dirs:
        print(f"Error: Reference model 'full_fine_tune' not found for dataset '{dataset_name}'. Skipping.")
        return

    model_names = list(model_dirs.keys())
    print(f"Selected folders for dataset '{dataset_name}':")
    for mode, info in model_dirs.items():
        folder_name = os.path.basename(info['folder_path'])
        print(f"  {mode}: {folder_name}")
    print(f"Found models: {model_names}")

    try:
        data_module_class = DATASET_CONFIG[dataset_name]['DataModule']
        data_module = data_module_class()
        data_module.prepare_data()
        data_module.setup(stage='test')
        dataloader = data_module.test_dataloader()
    except Exception as e:
        print(f"Error initializing data module for dataset '{dataset_name}': {e}. Skipping.")
        return

    # Instead of storing in memory, we will save paths to h5 files
    h5_files_dict = {}

    for model_name in model_names:
        model_info = model_dirs[model_name]
        model_dir = model_info['folder_path']
        mode_config = model_info['mode_config']

        run_folders = [
            d for d in os.listdir(model_dir)
            if os.path.isdir(os.path.join(model_dir, d)) and d.startswith('Run_')
        ]
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

        if dataset_name == 'DeepShip':
            num_classes = 4
        else:
            num_classes = 5

        args = SimpleNamespace(**mode_config)
        params = Parameters(args)

        model = load_model_with_args(model_path, model_name, num_classes, 0, params)
        if not model:
            continue

        # H5 file path for this model
        h5_file_path = os.path.join(features_base_dir, f"{dataset_name}_{model_name}_features.h5")
        
        try:
            total_samples, total_batches = extract_features(
                model, dataloader, device, h5_file_path, num_batches=num_batches
            )
            h5_files_dict[model_name] = h5_file_path
            print(f"Model '{model_name}': {total_samples} samples from {total_batches} batches used. Features saved to '{h5_file_path}'.")
        except Exception as e:
            print(f"Error processing model '{model_name}': {e}. Skipping.")

    # Compute similarity from h5 files
    if 'full_fine_tune' in h5_files_dict:
        output_plot_path = os.path.join(features_base_dir, f"{dataset_name}_cosine_similarity_plot.png")
        compute_layer_cosine_similarity(h5_files_dict, model_names, output_path=output_plot_path)
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
                print(f"Warning: Dataset '{dataset_name}' not recognized. Skipping folder '{entry}'.")
                continue
            if dataset_name not in dataset_folders:
                dataset_folders[dataset_name] = []
            dataset_folders[dataset_name].append(entry)
    return dataset_folders

def parse_args():
    parser = argparse.ArgumentParser(description='Run histogram experiments with mode-specific parameters')
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        nargs='+',
        choices=DATASET_CONFIG.keys(),
        help='Name of the dataset(s) to process. Choices are: ' + ', '.join(DATASET_CONFIG.keys())
    )
    return parser.parse_args()

def main():
    tb_logs_base_dir = 'tb_logs'
    features_base_dir = os.path.join('features', 'similarity_plots')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_batches = None

    args = parse_args()
    selected_datasets = args.dataset

    if not os.path.isdir(tb_logs_base_dir):
        print(f"Error: tb_logs directory '{tb_logs_base_dir}' does not exist.")
        return

    os.makedirs(features_base_dir, exist_ok=True)

    dataset_folders = traverse_tb_logs(tb_logs_base_dir)

    if selected_datasets:
        dataset_folders = {dataset: folders for dataset, folders in dataset_folders.items() if dataset in selected_datasets}
        missing_datasets = set(selected_datasets) - set(dataset_folders.keys())
        if missing_datasets:
            for ds in missing_datasets:
                print(f"Warning: Dataset '{ds}' not found in tb_logs. It will be skipped.")
    else:
        pass

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

    print("\nAll selected datasets processed.")

if __name__ == "__main__":
    main()
