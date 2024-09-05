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


class LitModel(L.LightningModule):

    def __init__(self, Params, model_name, num_classes, numBins, Dataset):
        super().__init__()

        self.learning_rate = Params['lr']


        self.model_ft, self.feature_extraction_layer = initialize_model(model_name, num_classes,
                                                                        numBins,Params['sample_rate'],
                                                                        spec_norm=Params['spec_norm'],
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
        self.test_acc = torchmetrics.classification.Accuracy(
            task="multiclass", num_classes=num_classes)
        
        self.save_hyperparameters()


    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        return y_pred

    def training_step(self, train_batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y = train_batch
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        loss = F.cross_entropy(y_pred, y)

        self.train_acc(y_pred, y)

        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        self.log('loss', loss, on_step=True, on_epoch=True)

        return loss


    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        val_loss = F.cross_entropy(y_pred, y)

        self.val_acc(y_pred, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)

        return val_loss

 
    def test_step(self, test_batch, batch_idx):
        x, y = test_batch

        y_feat = self.feature_extraction_layer(x)
        y_pred = self.model_ft(y_feat)
        
        test_loss = F.cross_entropy(y_pred, y)
        
        self.test_acc(y_pred, y)
    
        self.log('test_loss', test_loss, on_step=False, on_epoch=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True)

        return test_loss
    
    
    
    def configure_optimizers(self):
        # AdamW optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        
        # Cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
    
        return [optimizer], [scheduler]
    

    # def configure_optimizers(self):
    #     # Collect parameters for different parts of the model
    #     mlp_params = list(self.model_ft.mlp_head.parameters())
        
    #     histogram_params = []
    #     if self.h_mode == 'histogram':
    #         if self.h_location == 'after_encoder':
    #             histogram_params.extend(list(self.model_ft.histogram_layer.parameters()))
    #         elif self.h_location == 'within_each':
    #             histogram_params.extend(list(self.model_ft.histogram_layers.parameters()))
        
    #     adapters_params = []
    #     if self.Use_A:
    #         if self.a_location in ['both', 'mhsa']:
    #             adapters_params.extend(list(self.model_ft.adapters_mhsa.parameters()))
    #         if self.a_location in ['both', 'ffn']:
    #             adapters_params.extend(list(self.model_ft.adapters_ffn.parameters()))
        
    #     # Collect other parameters
    #     other_params = []
    #     for name, param in self.model_ft.named_parameters():
    #         if not param.requires_grad:
    #             continue
    #         if 'mlp_head' not in name and 'histogram_layer' not in name and 'histogram_layers' not in name and 'adapters_mhsa' not in name and 'adapters_ffn' not in name:
    #             other_params.append(param)
        
    #     # Define different learning rates
    #     mlp_lr = 1e-3
    #     histogram_lr = 1e-3
    #     adapters_lr = 1e-4
    #     other_lr = 1e-5
    
    #     # Create parameter groups with different learning rates
    #     optimizer_params = [
    #         {'params': mlp_params, 'lr': mlp_lr},
    #         {'params': adapters_params, 'lr': adapters_lr},
    #         {'params': other_params, 'lr': other_lr}
    #     ]
    
    #     if self.h_mode == 'histogram':
    #         optimizer_params.append({'params': histogram_params, 'lr': histogram_lr})
    
    #     optimizer = torch.optim.Adam(optimizer_params)
    #     return optimizer



