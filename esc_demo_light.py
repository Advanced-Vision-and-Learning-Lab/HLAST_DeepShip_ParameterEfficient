#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 11:01:06 2024

@author: amir.m
"""

from __future__ import print_function
from __future__ import division
import numpy as np
import argparse
import os
# PyTorch dependencies
import torch

# Local external libraries
from esc_Demo_Parameters import Parameters
from escDataModule import ESC50DataModule
from esc_LitModel import esc_LitModel
from esc50_utils import prepare_esc50_dataset


import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

# This code uses a newer version of numpy while other packages use an older version of numpy
# This is a simple workaround to avoid errors that arise from the deprecation of numpy data types
np.float = float  # module 'numpy' has no attribute 'float'
np.int = int  # module 'numpy' has no attribute 'int'
np.object = object  # module 'numpy' has no attribute 'object'
np.bool = bool  # module 'numpy' has no attribute 'bool'

import pdb
import shutil
def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(Params):
    # Model(s) to be used
    model_name = Params['Model_name']

    # Number of bins and input convolution feature maps after channel-wise pooling
    numBins = Params['numBins']

    h_mode = Params['histogram']
    h_shared = Params['histograms_shared']
    a_shared = Params['adapters_shared']
    
    batch_size = Params['batch_size']
    batch_size = batch_size['train']

    print('\nStarting Experiments...')
    
    original_data_dir = "./esc50_data"
    resampled_data_dir = "./esc50_data_resampled"
    target_sample_rate = 16000  # or any other desired sample rate

    # Prepare the dataset (download, extract, and resample if necessary)
    prepare_esc50_dataset(original_data_dir, resampled_data_dir, target_sample_rate)

    # Use the resampled dataset
    data_module = ESC50DataModule(data_dir=resampled_data_dir, batch_size=Params['batch_size'])
    data_module.setup()

    total_samples = len(data_module.train_dataloader().dataset)
    steps_per_epoch = total_samples // batch_size
    log_every_n_steps = max(1, steps_per_epoch // 10)  # Log at least 10 times per epoch

    torch.set_float32_matmul_precision('medium')
    all_val_accs = []
    numRuns = 3
    num_classes = 50  

    for run_number in range(numRuns):
        seed_everything(run_number + 1, workers=True)
        print(f'\nStarting Run {run_number}\n')

        for fold in range(3):  # 3-fold cross-validation
            data_module.set_fold(fold)

            checkpoint_callback = ModelCheckpoint(
                monitor='val_acc',
                filename=f'best-run{run_number}-fold{fold}-' + '{epoch:02d}-{val_acc:.2f}',
                save_top_k=1,
                mode='max',
                verbose=True,
                save_weights_only=True
            )

            early_stopping_callback = EarlyStopping(
                monitor='val_loss',
                patience=Params['patience'],
                verbose=True,
                mode='min'
            )

            model_AST = esc_LitModel(Params, model_name, num_classes, numBins)

            num_params = count_trainable_params(model_AST)
            print(f'Total Trainable Parameters: {num_params}\n')

            logger = TensorBoardLogger(
                save_dir=(
                    f"tb_logs/esc_{Params['feature']}_b{batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
                    f"_AdaptShared{a_shared}_{Params['adapter_location']}_{Params['adapter_mode']}_Hist{h_mode}Shared{h_shared}_{numBins}bins_{Params['histogram_location']}"
                    f"_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}_m{Params['number_mels']}/Run_{run_number}_Fold_{fold}"
                ),
                name="metrics"
            )

            trainer = L.Trainer(
                max_epochs=Params['num_epochs'],
                callbacks=[early_stopping_callback, checkpoint_callback],
                deterministic=False,
                logger=logger,
                log_every_n_steps=log_every_n_steps,
            )

            trainer.fit(model=model_AST, datamodule=data_module)

            best_val_acc = checkpoint_callback.best_model_score.item()
            all_val_accs.append(best_val_acc)

            results_filename = (
                f"tb_logs/esc_{Params['feature']}_b{batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
                f"_AdaptShared{a_shared}_{Params['adapter_location']}_{Params['adapter_mode']}_Hist{h_mode}Shared{h_shared}_{numBins}bins_{Params['histogram_location']}"
                f"_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}_m{Params['number_mels']}/Run_{run_number}_Fold_{fold}/metrics.txt"
            )

            with open(results_filename, "a") as file:
                file.write(f"Run_{run_number}_Fold_{fold}:\n\n")
                file.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")

    overall_avg_val_acc = np.mean(all_val_accs)
    overall_std_val_acc = np.std(all_val_accs)
     
    summary_filename = (
        f"tb_logs/esc_{Params['feature']}_b{batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
        f"_AdaptShared{a_shared}_{Params['adapter_location']}_{Params['adapter_mode']}_Hist{h_mode}Shared{h_shared}_{numBins}bins_{Params['histogram_location']}"
        f"_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}_m{Params['number_mels']}/summary_metrics.txt"
    )

    with open(summary_filename, "a") as file:
        file.write("Overall Results Across All Runs and Folds\n\n")
        file.write(f"Overall Average of Best Validation Accuracies: {overall_avg_val_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Validation Accuracies: {overall_std_val_acc:.4f}\n\n")

    


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run histogram experiments')
    parser.add_argument('--model', type=str, default='AST',
                        help='Select baseline model architecture')
    parser.add_argument('--histogram', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to use --no-histogram or --histogram')
    parser.add_argument('--histograms_shared', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use histogram shared')
    parser.add_argument('--adapters_shared', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use adapter shared')
    parser.add_argument('-numBins', type=int, default=16,
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
    parser.add_argument('--number_mels', type=int, default=64,
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
