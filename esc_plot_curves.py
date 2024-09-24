#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:38:19 2024

@author: amir.m
"""
from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import torch
from esc_Demo_Parameters import Parameters
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

np.float = float  # module 'numpy' has no attribute 'float'
np.int = int  # module 'numpy' has no attribute 'int'
np.object = object  # module 'numpy' has no attribute 'object'
np.bool = bool  # module 'numpy' has no attribute 'bool'


def extract_scalar_from_events(event_paths, scalar_name):
    scalar_values = []
    for event_path in event_paths:
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()
        if scalar_name in event_acc.Tags()['scalars']:
            scalar_events = event_acc.Scalars(scalar_name)
            values = [event.value for event in scalar_events]
            scalar_values.append(values)
    return scalar_values

def list_available_scalars(event_paths):
    available_scalars = set()
    for event_path in event_paths:
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()
        available_scalars.update(event_acc.Tags()['scalars'])
    return available_scalars

def save_learning_curves(log_dirs, output_path):
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5 colors for 5 folds
    line_styles = ['-', '--']
    line_width = 4
    
    for fold, log_dir in enumerate(log_dirs):
        event_paths = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith('events.out.tfevents.'):
                    event_paths.append(os.path.join(root, file))

        available_scalars = list_available_scalars(event_paths)
        print(f"Available Scalars for Fold {fold+1}:", available_scalars)
        
        train_loss = extract_scalar_from_events(event_paths, 'loss_epoch')
        val_loss = extract_scalar_from_events(event_paths, 'val_loss')

        if train_loss and val_loss:
            train_loss = train_loss[0] if isinstance(train_loss[0], list) else train_loss
            val_loss = val_loss[0] if isinstance(val_loss[0], list) else val_loss
            plt.plot(train_loss, label=f'Training Loss (Fold {fold+1})', color=colors[fold], linestyle=line_styles[0], lw=line_width)
            plt.plot(val_loss, label=f'Validation Loss (Fold {fold+1})', color=colors[fold], linestyle=line_styles[1], lw=line_width)

    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.title('Learning Curves (5-Fold CV)', fontsize=18)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.8)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_accuracy_curves(log_dirs, output_path):
    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5 colors for 5 folds
    line_styles = ['-', '--']
    line_width = 4
    
    for fold, log_dir in enumerate(log_dirs):
        event_paths = []
        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if file.startswith('events.out.tfevents.'):
                    event_paths.append(os.path.join(root, file))

        available_scalars = list_available_scalars(event_paths)
        print(f"Available Scalars for Fold {fold+1}:", available_scalars)
        
        train_acc = extract_scalar_from_events(event_paths, 'train_acc')
        val_acc = extract_scalar_from_events(event_paths, 'val_acc')

        if train_acc and val_acc:
            train_acc = train_acc[0] if isinstance(train_acc[0], list) else train_acc
            val_acc = val_acc[0] if isinstance(val_acc[0], list) else val_acc
            plt.plot(train_acc, label=f'Training Accuracy (Fold {fold+1})', color=colors[fold], linestyle=line_styles[0], lw=line_width)
            plt.plot(val_acc, label=f'Validation Accuracy (Fold {fold+1})', color=colors[fold], linestyle=line_styles[1], lw=line_width)

    plt.xlabel('Epochs', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.title('Accuracy Curves (5-Fold CV)', fontsize=18)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.8)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def main(Params):
    torch.set_float32_matmul_precision('medium')
    
    num_runs = 3  # Number of runs

    for run in range(num_runs):
        output_dir = f"features/Curves_esc_{Params['feature']}_b{Params['batch_size']['train']}_{Params['sample_rate']}_{Params['train_mode']}_AdaptShared{Params['adapters_shared']}_{Params['adapter_location']}_{Params['adapter_mode']}_Hist{Params['histogram']}Shared{Params['histograms_shared']}_{Params['numBins']}bins_{Params['histogram_location']}_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}_m{Params['number_mels']}/Run_{run}"
        os.makedirs(output_dir, exist_ok=True)
        
        log_dirs = []
        for fold in range(1, 6):  
            log_dir = f"tb_logs/esc_{Params['feature']}_b{Params['batch_size']['train']}_{Params['sample_rate']}_{Params['train_mode']}_AdaptShared{Params['adapters_shared']}_{Params['adapter_location']}_{Params['adapter_mode']}_Hist{Params['histogram']}Shared{Params['histograms_shared']}_{Params['numBins']}bins_{Params['histogram_location']}_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}_m{Params['number_mels']}/Run_{run}_Fold_{fold}/metrics"
            log_dirs.append(log_dir)
        
        # Save learning curves
        learning_curves_output_path = os.path.join(output_dir, f"learning_curves_run{run}.png")
        save_learning_curves(log_dirs, learning_curves_output_path)
        
        # Save accuracy curves
        accuracy_curves_output_path = os.path.join(output_dir, f"accuracy_curves_run{run}.png")
        save_accuracy_curves(log_dirs, accuracy_curves_output_path)
        
def parse_args():
    parser = argparse.ArgumentParser(
        description='Run histogram experiments')
    parser.add_argument('--model', type=str, default='AST',
                        help='Select baseline model architecture')
    parser.add_argument('--histogram', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to use --no-histogram or --histogram')
    parser.add_argument('--histograms_shared', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to use histogram shared')
    parser.add_argument('--adapters_shared', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to use adapter shared')
    parser.add_argument('-numBins', type=int, default=8,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--train_mode', type=str, default='full_fine_tune',
                        help='full_fine_tune or linear_probing or adapters')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=64,
                        help='input batch size for validation (default: 512)')
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
    parser.add_argument('--patience', type=int, default=25,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--window_length', type=int, default=4096,
                        help='window length')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='hop length')
    parser.add_argument('--number_mels', type=int, default=128,
                        help='number of mels')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Dataset Sample Rate'),
    parser.add_argument('--adapter_location', type=str, default='None',
                        help='Location for the adapter layers (default: ffn)')
    parser.add_argument('--adapter_mode', type=str, default='None',
                        help='Mode for the adapter layers (default: parallel)')
    parser.add_argument('--histogram_location', type=str, default='None',
                        help='Location for the histogram layers (default: ffn)')
    parser.add_argument('--histogram_mode', type=str, default='None',
                        help='Mode for the histogram layers (default: parallel)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)

