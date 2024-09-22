#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:10:34 2024

@author: amir.m
"""

from __future__ import print_function
from __future__ import division

# PyTorch dependencies
import torch

# Local external libraries
from Utils.Network_functions import initialize_model

import torch.nn.functional as F
import lightning as L
import torchmetrics
import pdb

class esc_LitModel(L.LightningModule):

    def __init__(self, Params, model_name, num_classes, numBins):
        super().__init__()

        self.learning_rate = Params['lr']


        self.model_ft, self.feature_extraction_layer = initialize_model(model_name, num_classes,
                                                                        numBins,Params['sample_rate'],
                                                                        window_length=Params['window_length'],
                                                                        hop_length=Params['hop_length'],
                                                                        number_mels=Params['number_mels'],
                                                                        t_mode=Params['train_mode'],
                                                                        histogram=Params['histogram'],
                                                                        h_shared=Params['histograms_shared'],
                                                                        a_shared=Params['adapters_shared'],
                                                                        parallel=Params['parallel'],
                                                                        use_pretrained=Params['use_pretrained'],
                                                                        input_feature=Params['feature'],
                                                                        adapter_location=Params['adapter_location'],
                                                                        adapter_mode=Params['adapter_mode'],
                                                                        histogram_location=Params['histogram_location'],
                                                                        histogram_mode=Params['histogram_mode'])



        self.train_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)

        self.save_hyperparameters()

        # Print number of trainable parameters for each component
        self.print_parameter_counts()
    
    def print_parameter_counts(self):
        def count_parameters(params):
            return sum(p.numel() for p in params if p.requires_grad)
    
        # MLP head parameters
        if hasattr(self.model_ft, 'mlp_head'):
            mlp_head_params = self.model_ft.mlp_head.parameters()
            print(f"Number of trainable parameters in MLP head: {count_parameters(mlp_head_params)}")
    
        # Histogram layers parameters
        if self.model_ft.use_histogram:
            histogram_params = self.get_histogram_parameters()
            if histogram_params:
                print(f"Number of trainable parameters in histogram layers: {count_parameters(histogram_params)}")
    
        # Adapter parameters
        if self.model_ft.use_adapters:
            adapter_params = self.get_adapter_parameters()
            if adapter_params:
                print(f"Number of trainable parameters in adapters: {count_parameters(adapter_params)}")
    
        # Other parameters
        other_params = self.get_other_parameters()
        if other_params:
            print(f"Number of trainable parameters in others: {count_parameters(other_params)}")


    def preprocess_input(self, x):
        # Reshape from [batch_size, channels, time] to [batch_size, time]
        return x.squeeze(1)

    def forward(self, x):
        x = self.preprocess_input(x)
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        return y_pred

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        x = self.preprocess_input(x)
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        loss = F.cross_entropy(y_pred, y)

        self.train_acc(y_pred, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        x = self.preprocess_input(x)
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        val_loss = F.cross_entropy(y_pred, y)

        self.val_acc(y_pred, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        return val_loss

    
    # def configure_optimizers(self):
    #     # AdamW optimizer
    #     optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
    #     # Cosine annealing scheduler
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
    
    #     return [optimizer], [scheduler]
    
    def configure_optimizers(self):
        # Define different learning rates
        base_lr = self.learning_rate  # Learning rate for the rest of the model
        mlp_head_lr = self.learning_rate * 10  # Higher learning rate for mlp_head 
    
        # Separate the parameters
        optimizer = torch.optim.AdamW([
            {'params': self.model_ft.mlp_head.parameters(), 'lr': mlp_head_lr},  # Higher LR for mlp_head
            {'params': [p for n, p in self.named_parameters() if "mlp_head" not in n], 'lr': base_lr}  # Base LR for the rest
        ])
    
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
    
        return [optimizer], [scheduler]



                
    def get_histogram_parameters(self):
        params = []
        if self.model_ft.histogram_location in ['all', 'mhsa_ffn','mhsa_out', 'mhsa']:
            params += list(self.model_ft.histogram_layers_mhsa.parameters())
        if self.model_ft.histogram_location in ['all', 'mhsa_ffn', 'ffn_out', 'ffn']:
            params += list(self.model_ft.histogram_layers_ffn.parameters())
        if self.model_ft.histogram_location in ['all', 'mhsa_out', 'ffn_out', 'out']:
            params += list(self.model_ft.histogram_layers_out.parameters())
        return params

    
    def get_adapter_parameters(self):
        params = []
        if self.model_ft.adapter_location in ['all', 'mhsa_ffn','mhsa_out', 'mhsa']:
            params += list(self.model_ft.adapters_mhsa.parameters())
        if self.model_ft.adapter_location in ['all', 'mhsa_ffn', 'ffn_out', 'ffn']:
            params += list(self.model_ft.adapters_ffn.parameters())
        if self.model_ft.adapter_location in ['all', 'mhsa_out', 'ffn_out', 'out']:
            params += list(self.model_ft.adapters_out.parameters())
        return params

    
    def get_other_parameters(self):
        # Gather other parameters that don't belong to MLP head, histogram, or adapters
        other_params = []
        for name, param in self.model_ft.named_parameters():

            if 'mlp_head' not in name and 'histogram' not in name and 'adapter' not in name:
                other_params.append(param)
                
        for name, param in self.feature_extraction_layer.named_parameters():
                other_params.append(param)
                        
        return other_params
