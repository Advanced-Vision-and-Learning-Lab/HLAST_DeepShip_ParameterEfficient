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

# PyTorch dependencies
import torch

# Local external libraries
from Demo_Parameters import Parameters
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from Datasets.Get_preprocessed_data import process_data

# This code uses a newer version of numpy while other packages use an older version of numpy
# This is a simple workaround to avoid errors that arise from the deprecation of numpy data types
np.float = float  # module 'numpy' has no attribute 'float'
np.int = int  # module 'numpy' has no attribute 'int'
np.object = object  # module 'numpy' has no attribute 'object'
np.bool = bool  # module 'numpy' has no attribute 'bool'

from SSDataModule import SSAudioDataModule
from LitModel import LitModel

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(Params):

    # Name of dataset
    Dataset = Params['Dataset']
    
    # Model(s) to be used
    model_name = Params['Model_name']

    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]

    # Number of runs and/or splits for dataset
    numRuns = Params['Splits'][Dataset]

    # Number of bins and input convolution feature maps after channel-wise pooling
    numBins = Params['numBins']

    h_mode = Params['histogram']
    h_shared = Params['histograms_shared']
    a_shared = Params['adapters_shared']
    
    batch_size = Params['batch_size']
    batch_size = batch_size['train']

    print('\n\n\nStarting Experiments...')
    
    run_number = 0
    seed_everything(run_number+1, workers=True)
        
    new_dir = Params["new_dir"] 
    process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
    data_module = SSAudioDataModule(new_dir, batch_size=batch_size, sample_rate=Params['sample_rate'])
    data_module.prepare_data()
    
    torch.set_float32_matmul_precision('medium')
    all_val_accs = []
    all_test_accs = []
    for run_number in range(0, numRuns):
        
        if run_number != 0:
            seed_everything(run_number+1, workers=True)
                 
        print(f'\nStarting Run {run_number}\n')
    
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            filename='best-{epoch:02d}-{val_acc:.2f}',
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


        model_AST = LitModel(Params, model_name, num_classes, numBins, Dataset)

        num_params = count_trainable_params(model_AST)
        print(f'Total Trainable Parameters: {num_params}\n')
        
        
        
        # logger = TensorBoardLogger(
        #     save_dir = (f"tb_logs/{Params['feature']}_b{batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
        #    f"_AdaptShared{a_shared}_{Params['adapter_location']}_{Params['adapter_mode']}_Hist{h_mode}Shared{h_shared}_{numBins}bins_{Params['histogram_location']}"
        #    f"_{Params['histogram_mode']}/Run_{run_number}"),
        #     name="metrics"
        # )

        logger = TensorBoardLogger(
            save_dir=(
                f"tb_logs/{Params['feature']}_b{batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
                f"_AdaptShared{a_shared}_{Params['adapter_location']}_{Params['adapter_mode']}_Hist{h_mode}Shared{h_shared}_{numBins}bins_{Params['histogram_location']}"
                f"_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}_m{Params['number_mels']}/Run_{run_number}"
            ),
            name="metrics"
        )


        trainer = L.Trainer(
            max_epochs=Params['num_epochs'],
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=False,
            logger=logger,
        )

        trainer.fit(model=model_AST, datamodule=data_module)
    
        best_val_acc = checkpoint_callback.best_model_score.item()
        all_val_accs.append(best_val_acc)
    
        best_model_path = checkpoint_callback.best_model_path
        best_model = LitModel.load_from_checkpoint(
            checkpoint_path=best_model_path,
            Params=Params,
            model_name=model_name,
            num_classes=num_classes,
            Dataset=Dataset,
            pretrained_loaded=True,
            run_number=run_number
        )
    
        test_results = trainer.test(model=best_model, datamodule=data_module)
        
        best_test_acc = test_results[0]['test_acc']
        all_test_accs.append(best_test_acc)
    
    
        results_filename = (
        f"tb_logs/{Params['feature']}_b{batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
        f"_AdaptShared{a_shared}_{Params['adapter_location']}_{Params['adapter_mode']}_Hist{h_mode}Shared{h_shared}_{numBins}bins_{Params['histogram_location']}"
        f"_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}_m{Params['number_mels']}/Run_{run_number}/metrics.txt"
        )
        #results_filename = (f"tb_logs/{Params['feature']}_b{batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
        #                    f"_AdaptShared{a_shared}_{Params['adapter_location']}_{Params['adapter_mode']}_Hist{h_mode}Shared{h_shared}_{numBins}bins_{Params['histogram_location']}"
        #                    f"_{Params['histogram_mode']}/Run_{run_number}/metrics.txt")
        with open(results_filename, "a") as file:
            file.write(f"Run_{run_number}:\n\n")
            file.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
            file.write(f"Best Test Accuracy: {best_test_acc:.4f}\n\n")
    
    overall_avg_val_acc = np.mean(all_val_accs)
    overall_std_val_acc = np.std(all_val_accs)
    
    overall_avg_test_acc = np.mean(all_test_accs)
    overall_std_test_acc = np.std(all_test_accs)
    
    
    
    summary_filename = (
        f"tb_logs/{Params['feature']}_b{batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
        f"_AdaptShared{a_shared}_{Params['adapter_location']}_{Params['adapter_mode']}_Hist{h_mode}Shared{h_shared}_{numBins}bins_{Params['histogram_location']}"
        f"_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}_m{Params['number_mels']}/summary_metrics.txt"
    )

    #summary_filename = (f"tb_logs/{Params['feature']}_b{batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
    #                f"_AdaptShared{a_shared}_{Params['adapter_location']}_{Params['adapter_mode']}_Hist{h_mode}Shared{h_shared}_{numBins}bins_{Params['histogram_location']}"
    #                f"_{Params['histogram_mode']}/summary_metrics.txt")
    
    with open(summary_filename, "a") as file:
        file.write("Overall Results Across All Runs\n\n")
        file.write(f"Overall Average of Best Validation Accuracies: {overall_avg_val_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Validation Accuracies: {overall_std_val_acc:.4f}\n\n")
        file.write(f"Overall Average of Best Test Accuracies: {overall_avg_test_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Test Accuracies: {overall_std_test_acc:.4f}\n\n")

    


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
    parser.add_argument('--data_selection', type=int, default=0,
                        help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('--train_mode', type=str, default='full_fine_tune',
                        help='full_fine_tune or linear_probing or adapters')
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
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
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
