#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 19:38:19 2024

@author: amir.m
"""
from __future__ import print_function
from __future__ import division

import matplotlib.pyplot as plt
from itertools import cycle
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
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import itertools    

def save_avg_confusion_matrix(model_paths, test_loader, class_names, device, output_path):
    all_cms = []

    for path in model_paths:
        model = LitModel.load_from_checkpoint(checkpoint_path=path)
        model.eval()
        model.to(device)

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                x = x.to(device)
                y = y.to(device)
                y_pred = model(x)
                all_preds.append(y_pred.cpu().argmax(dim=1))
                all_labels.append(y.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        # Compute confusion matrix for the current run
        cm = confusion_matrix(all_labels, all_preds)
        all_cms.append(cm)

    # Convert to numpy array for easier manipulation
    all_cms = np.array(all_cms)

    # Compute mean and standard deviation of confusion matrices
    mean_cm = np.mean(all_cms, axis=0)
    std_cm = np.std(all_cms, axis=0)
    mean_cm_percent = 100 * mean_cm.astype('float') / mean_cm.sum(axis=1)[:, np.newaxis]
    std_cm_percent = 100 * std_cm.astype('float') / (std_cm.sum(axis=1)[:, np.newaxis] + 1e-6)

    # Plot average confusion matrix with percentages and standard deviations
    plt.figure(figsize=(6, 6))  # Increased figure size
    ax = sns.heatmap(mean_cm_percent, annot=False, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=True, annot_kws={"size": 14}, vmin=0, vmax=100)

    # Increase font size for colorbar values
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    
    for i, j in itertools.product(range(mean_cm.shape[0]), range(mean_cm.shape[1])):
        text_value = f"{int(mean_cm[i, j])} ± {int(std_cm[i, j])}\n({mean_cm_percent[i, j]:.1f} ±\n {std_cm_percent[i, j]:.1f}%)"

        font_size = 14 if len(text_value) < 30 else 12  # Adjust size based on content length
        ax.text(j + 0.5, i + 0.5, text_value, ha="center", va="center", fontsize=font_size,
                color="white" if mean_cm_percent[i, j] > 50 else "black")

    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Average Confusion Matrix', fontsize=14)
    plt.xticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, fontsize=12)
    plt.yticks(ticks=np.arange(len(class_names)) + 0.5, labels=class_names, fontsize=12)
    plt.tight_layout()

    # Save the figure
    plt.savefig(output_path, dpi=300)
    plt.close()

    


import glob
from LitModel import LitModel

def main(Params):
    # Name of dataset
    Dataset = Params['Dataset']

    # Model(s) to be used
    model_name = Params['Model_name']

    # Number of classes in dataset
    num_classes = Params['num_classes'][Dataset]

    batch_size = Params['batch_size']
    batch_size = batch_size['train']

    print('\nStarting Experiments...')
    
    numRuns = 3
    s_rate = Params['sample_rate']
    new_dir = Params["new_dir"]
    
    print("\nDataset sample rate: ", s_rate)
    print("\nModel name: ", model_name, "\n")
    
    data_module = SSAudioDataModule(new_dir, batch_size=batch_size, sample_rate=s_rate)
    data_module.prepare_data()

    torch.set_float32_matmul_precision('medium')
    
    # Collecting model paths for average confusion matrix calculation
    model_paths = []
    for run_number in range(numRuns):
        best_model_path = glob.glob(f"tb_logs/STFT_b64_16bins_16000_linear_probing_HistTrue_Adapt_None_Adapt_None_Hist_all_Hist_parallel/Run_{run_number}/metrics/version_0/checkpoints/*.ckpt")[0]
        model_paths.append(best_model_path)

    # Use the first run's model for individual plots
    best_model_path = model_paths[0]
    run_number = 0

    # best_model = LitModel.load_from_checkpoint(
    #     checkpoint_path=best_model_path,
    #     Params=Params,
    #     model_name=model_name,
    #     num_classes=num_classes,
    #     Dataset=Dataset,
    #     pretrained_loaded=True,
    #     run_number=run_number
    # )

    # Get the test dataloader from the data module
    test_loader = data_module.test_dataloader()
    
    # Extract class names dynamically from the class_to_idx dictionary
    class_names = list(data_module.class_to_idx.keys())

    # Create directory if it doesn't exist
    output_dir = f"features/Curves_{Params['feature']}_{s_rate}_Run{run_number}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output path for the ROC plot
    #roc_output_path = os.path.join(output_dir, "roc_curve.png")
    
    # Plot ROC curves and save the figure
    #plot_multiclass_roc(best_model, test_loader, class_names, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path=roc_output_path)
        
    # Define output path for the confusion matrix plot with percentages
    #cm_prc_output_path = os.path.join(output_dir, "confusion_matrix_prc.png")
        
    # Save confusion matrix with percentages
    #save_confusion_matrix_with_percentage(best_model, test_loader, class_names, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path=cm_prc_output_path)
    
    # Define output path for the average confusion matrix plot
    avg_cm_output_path = os.path.join(output_dir, "avg_confusion_matrix.png")
    
    # Save average confusion matrix across runs
    save_avg_confusion_matrix(model_paths, test_loader, class_names, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), output_path=avg_cm_output_path)
      
    # Define output path for the learning curves plot
    learning_curves_output_path = os.path.join(output_dir, "learning_curves.png")
    

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
    parser.add_argument('--val_batch_size', type=int, default=128,
                        help='input batch size for validation (default: 512)')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', default=True, action=argparse.BooleanOptionalAction,
                        help='enables CUDA training')
    parser.add_argument('--audio_feature', type=str, default='STFT',
                        help='Audio feature for extraction')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='Select optimizer')
    parser.add_argument('--patience', type=int, default=5,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Dataset Sample Rate'),
    parser.add_argument('--adapter_location', type=str, default='None',
                        help='Location for the adapter layers (default: ffn)')
    parser.add_argument('--adapter_mode', type=str, default='None',
                        help='Mode for the adapter layers (default: parallel)')
    parser.add_argument('--histogram_location', type=str, default='mhsa',
                        help='Location for the histogram layers (default: ffn)')
    parser.add_argument('--histogram_mode', type=str, default='parallel',
                        help='Mode for the histogram layers (default: parallel)')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)
