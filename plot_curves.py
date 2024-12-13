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
from Demo_Parameters import Parameters
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalar_from_events(event_paths, scalar_name):
    scalar_values = []
    for event_path in event_paths:
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()
        if scalar_name in event_acc.Tags()['scalars']:
            scalar_events = event_acc.Scalars(scalar_name)
            values = [event.value for event in scalar_events]
            scalar_values.extend(values)
    return scalar_values


def list_available_scalars(event_paths):
    available_scalars = set()
    for event_path in event_paths:
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()
        available_scalars.update(event_acc.Tags()['scalars'])
    return available_scalars


def save_learning_curves(log_dir, output_path):
    # Get all event files in the log directory
    event_paths = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents.'):
                event_paths.append(os.path.join(root, file))

    # List all available scalars
    available_scalars = list_available_scalars(event_paths)
    print("Available Scalars:", available_scalars)
    train_loss = extract_scalar_from_events(event_paths, 'loss_epoch')
    val_loss = extract_scalar_from_events(event_paths, 'val_loss')

    if train_loss and val_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(train_loss, label='Training Loss', color='blue', lw=3)
        plt.plot(val_loss, label='Validation Loss', color='orange', lw=3)
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.title('Learning Curves', fontsize=18)
        plt.legend(loc="best", fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        print("Required scalars ('loss_epoch', 'val_loss') not found in event files.")


def save_accuracy_curves(log_dir, output_path):
    # Get all event files in the log directory
    event_paths = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents.'):
                event_paths.append(os.path.join(root, file))

    # List all available scalars
    available_scalars = list_available_scalars(event_paths)
    print("Available Scalars:", available_scalars)
    train_acc = extract_scalar_from_events(event_paths, 'train_acc')
    val_acc = extract_scalar_from_events(event_paths, 'val_acc')

    # Plot accuracy curves if both scalars are available
    if train_acc and val_acc:
        plt.figure(figsize=(6, 4))
        plt.plot(train_acc, label='Training Accuracy', color='green', lw=3)
        plt.plot(val_acc, label='Validation Accuracy', color='red', lw=3)
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.title('Accuracy Curves', fontsize=18)
        plt.legend(loc="best", fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        print("Required scalars ('train_acc', 'val_acc') not found in event files.")


def main(Params):

    torch.set_float32_matmul_precision('medium')
    
    run_number = 0

    output_dir = f"features/Curves/LogMelFBank_b64_16000_adapters_AdaptSharedTrue_64_mhsa_parallel_SharedTrue_16bins_None_None_w1024_h512_m64/Run_{run_number}"
    os.makedirs(output_dir, exist_ok=True)
    
    log_dir = f'tb_logs/LogMelFBank_b64_16000_adapters_AdaptSharedTrue_64_mhsa_parallel_SharedTrue_16bins_None_None_w1024_h512_m64/Run_{run_number}/metrics'
    
    # Save learning curves
    learning_curves_output_path = os.path.join(output_dir, "learning_curves.png")
    save_learning_curves(log_dir, learning_curves_output_path)
    
    # Save accuracy curves
    accuracy_curves_output_path = os.path.join(output_dir, "accuracy_curves.png")
    save_accuracy_curves(log_dir, accuracy_curves_output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run histogram experiments')
    parser.add_argument('--model', type=str, default='AST',
                        help='Select baseline model architecture')
    parser.add_argument('--histograms_shared', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use histogram shared')
    parser.add_argument('--adapters_shared', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use adapter shared')
    parser.add_argument('--data_selection', type=int, default=1,
                        help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('-RR', type=int, default=128,
                        help='Adapter Reduction Rate (default: 128)')
    parser.add_argument('--train_mode', type=str, default='full_fine_tune',
                        help='full_fine_tune or linear_probing or adapters or histogram')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=64,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--audio_feature', type=str, default='LogMelFBank',
                        help='Audio feature for extraction')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Select optimizer')
    parser.add_argument('--patience', type=int, default=25,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--window_length', type=int, default=1024,
                        help='window length')
    parser.add_argument('--hop_length', type=int, default=1000,
                        help='hop length')
    parser.add_argument('--number_mels', type=int, default=64,
                        help='number of mels')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Dataset Sample Rate'),
    parser.add_argument('--segment_length', type=int, default=5,
                        help='Dataset Segment Length'),
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
