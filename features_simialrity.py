#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:43:43 2024

@author: amir.m
"""

from __future__ import print_function
from __future__ import division

import numpy as np
import os

import argparse
import torch

from Demo_Parameters import Parameters

np.float = float  # module 'numpy' has no attribute 'float'
np.int = int  # module 'numpy' has no attribute 'int'
np.object = object  # module 'numpy' has no attribute 'object'
np.bool = bool  # module 'numpy' has no attribute 'bool'

from SSDataModule import SSAudioDataModule
import glob
from LitModel import LitModel
import pdb
from functools import partial


# Define a dictionary to store the outputs
layer_outputs = {}

# Hook function to capture layer outputs
def hook_fn(layer_name, module, input, output):
    layer_outputs[layer_name] = output

# Register hooks for each layer
def register_hooks(model):
    hooks = []
    # Iterate over each block in the model
    for i, blk in enumerate(model.model_ft.v.blocks):
        # Register a hook for the mlp module in each block
        hook = blk.mlp.drop.register_forward_hook(partial(hook_fn, f'block_{i}_mlp'))
        hooks.append(hook)
    return hooks

def extract_features(model, dataloader, device, num_batches=None):
    model.to(device)
    model.eval()
    features = []
    layer_outputs.clear()  # Clear previous outputs

    hooks = register_hooks(model)  # Register hooks

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_batches is not None and i >= num_batches:
                break

            inputs, _ = batch
            inputs = inputs.to(device)
            _ = model(inputs)  # Forward pass to trigger hooks
            
            # Collect features for each layer separately
            batch_features = [value.clone() for key, value in layer_outputs.items()]
            features.append(batch_features)

    # Remove hooks after extracting features
    for hook in hooks:
        hook.remove()

    sample_features = features[0] if features else []
    num_layers = len(sample_features)
    
    print("Number of layers captured:", num_layers)
    if num_layers > 0:
        print("Shape:", sample_features[0].shape)
    
    return features


from sklearn.metrics.pairwise import cosine_similarity

def compute_layer_cosine_similarity(features_dict, model_names):
    """
    Compute cosine similarity of feature maps for each layer compared to the full fine-tune model.
    """
    cosine_similarities = {i: [] for i in range(12)}
    
    # Assume the first model in the list (full_fine_tune) is the reference model
    reference_features = features_dict[model_names[0]]
    
    for model_name in model_names:
        features = features_dict[model_name]
        for layer_index in range(12):
            # concatenate the features from all batches for the current layer
            ref_concat = torch.cat([f[layer_index].view(-1) for f in reference_features], dim=0).cpu().numpy()
            comp_concat = torch.cat([f[layer_index].view(-1) for f in features], dim=0).cpu().numpy()
            
            # Compute cosine similarity
            cos_sim = cosine_similarity(ref_concat.reshape(1, -1), comp_concat.reshape(1, -1))
            cosine_similarities[layer_index].append(cos_sim.item())
    
    return cosine_similarities
def save_cosine_similarity(cosine_similarities, model_names, filename):
    """
    Save cosine similarity results for each layer across models.
    """
    with open(filename, 'w') as file:
        for layer_index, similarities in cosine_similarities.items():
            sim_str = ', '.join([f"{model_names[i]}: {similarity:.6f}" for i, similarity in enumerate(similarities)])
            file.write(f"Layer {layer_index} - Cosine Similarities: [{sim_str}]\n")
            file.write("\n")

from scipy.stats import shapiro
def test_normality(features_dict, model_names):
    """
    Perform Shapiro-Wilk test for normality for each layer.
    """
    normality_results = {i: [] for i in range(12)}
    
    for model_name in model_names:
        features = features_dict[model_name]
        for layer_index in range(12):
            # concatenate the features from all batches for the current layer
            concat_features = torch.cat([f[layer_index].view(-1) for f in features], dim=0).cpu().numpy()
            
            # Shapiro-Wilk test
            stat, p_value = shapiro(concat_features)
            normality_results[layer_index].append(p_value)
    
    return normality_results

def save_normality_results(normality_results, model_names, filename):
    """
    Save Shapiro-Wilk normality test results for each layer across models.
    """
    with open(filename, 'w') as file:
        for layer_index, p_values in normality_results.items():
            p_val_str = ', '.join([f"{model_names[i]}: {p_value:.6f}" for i, p_value in enumerate(p_values)])
            file.write(f"Layer {layer_index} - Normality Test (p-values): [{p_val_str}]\n")
            file.write("\n")
            
def main(Params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Name of dataset
    Dataset = Params['Dataset']
    
    # Model(s) to be used
    model_name = Params['Model_name']

    # Number of bins and input convolution feature maps after channel-wise pooling
    numBins = Params['numBins']
    
    batch_size = Params['batch_size']
    batch_size = batch_size['train']
    
    run_number = 0        
    new_dir = Params["new_dir"] 
    print("\nModel name: ", model_name, "\n")
    
    data_module = SSAudioDataModule(new_dir, batch_size=batch_size, sample_rate=Params['sample_rate'])
    data_module.prepare_data()
    
    torch.set_float32_matmul_precision('medium')
    

    # Load models
    model_paths = {
        'full_fine_tune': glob.glob(f"tb_logs/LogMelFBank_b64_16000_full_fine_tune_AdaptSharedFalse_None_None_HistFalseSharedFalse_16bins_None_None/Run_{run_number}/metrics/version_0/checkpoints/*.ckpt")[0],
        'linear_probing': glob.glob(f"tb_logs/LogMelFBank_b64_16000_linear_probing_AdaptSharedFalse_None_None_HistFalseSharedTrue_16bins_None_None/Run_{run_number}/metrics/version_0/checkpoints/*.ckpt")[0],
        'histogram': glob.glob(f"tb_logs/LogMelFBank_b64_16000_linear_probing_AdaptSharedFalse_None_None_HistTrueSharedTrue_8bins_mhsa_parallel/Run_{run_number}/metrics/version_0/checkpoints/*.ckpt")[0],
        'adapters': glob.glob(f"tb_logs/128LogMelFBank_b64_16000_adapters_AdaptSharedTrue_mhsa_parallel_HistFalseSharedFalse_16bins_None_None/Run_{run_number}/metrics/version_0/checkpoints/*.ckpt")[0]
    }
    
    
    # Define model-specific parameters
    model_params = {
        'full_fine_tune': {
            'adapter_location': 'None',
            'adapter_mode': 'None',
            'histogram_location': 'None',
            'histogram_mode': 'None',
            'train_mode': 'full_fine_tune'
        },
        'linear_probing': {
            'adapter_location': 'None',
            'adapter_mode': 'None',
            'histogram_location': 'None',
            'histogram_mode': 'None',
            'train_mode': 'linear_probing'
        },
        'histogram': {
            'adapter_location': 'None',
            'adapter_mode': 'None',
            'histogram_location': 'mhsa',
            'histogram_mode': 'parallel',
            'train_mode': 'linear_probing'
        },
        'adapters': {
            'adapter_location': 'mhsa',
            'adapter_mode': 'parallel',
            'histogram_location': 'None',
            'histogram_mode': 'None',
            'train_mode': 'adapters'
        }
    }
    
    def load_model(name, path, model_params):
        # Load model with specific parameters
        params = Params.copy()
        params.update(model_params[name])
        
        return LitModel.load_from_checkpoint(
            checkpoint_path=path,
            Params=params,
            model_name=model_name,
            numBins=numBins,
            Dataset=Dataset
        )
    
    # Load models with specific parameters
    models = {
        name: load_model(name, path, model_params)
        for name, path in model_paths.items()
    }

    # Get the test dataloader
    test_loader = data_module.test_dataloader()
    
    # Extract features
    num_batches = 4
    features_dict = {}
    for name, model in models.items():
        features = extract_features(model, test_loader, device, num_batches=num_batches)
        features_dict[name] = features
    

    # Compute cosine similarity across models
    cosine_similarities = compute_layer_cosine_similarity(features_dict, list(models.keys()))
    
    # Save cosine similarity results using the model names
    save_cosine_similarity(cosine_similarities, list(models.keys()), f'features/logmel_cosine_similarity_results_{num_batches}.txt')
 
    # Test for normality across models
    normality_results = test_normality(features_dict, list(models.keys()))
    
    # Save normality test results
    save_normality_results(normality_results, list(models.keys()), f'features/logmel_normality_test_results_{num_batches}.txt')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run histogram experiments')
    parser.add_argument('--model', type=str, default='AST',
                        help='Select baseline model architecture')
    parser.add_argument('--histogram', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use --no-histogram or --histogram')
    parser.add_argument('--histograms_shared', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use histogram shared')
    parser.add_argument('--adapters_shared', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use adapter shared')
    parser.add_argument('--data_selection', type=int, default=0,
                        help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--train_mode', type=str, default='linear_probing',
                        help='full_fine_tune or linear_probing or adapters')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=64,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
    parser.add_argument('--audio_feature', type=str, default='LogMelFBank',
                        help='Audio feature for extraction')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Select optimizer')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Dataset Sample Rate'),
    parser.add_argument('--spec_norm', type=bool, default=False,
                        help='Normalize spectrograms')
    parser.add_argument('--adapter_location', type=str, default='mhsa_ffn',
                        help='Location for the adapter layers (default: ffn)')
    parser.add_argument('--adapter_mode', type=str, default='parallel',
                        help='Mode for the adapter layers (default: parallel)')
    parser.add_argument('--histogram_location', type=str, default='mhsa_ffn',
                        help='Location for the histogram layers (default: ffn)')
    parser.add_argument('--histogram_mode', type=str, default='parallel',
                        help='Mode for the histogram layers (default: parallel)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)
