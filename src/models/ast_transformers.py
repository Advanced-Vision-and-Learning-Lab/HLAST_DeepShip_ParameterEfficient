#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:42:24 2024

@author: amir.m
"""

import torch
import torch.nn as nn
from transformers import ASTModel as TransformersASTModel, ASTConfig

class ASTModel(nn.Module):
    def __init__(self, label_dim=527, model_size='base384', use_pretrained=True):
        super(ASTModel, self).__init__()
        
        # Load the AST configuration
        config = ASTConfig.from_pretrained(model_size)
        config.num_labels = label_dim
        
        # Initialize the AST model from transformers
        self.model = TransformersASTModel.from_pretrained(model_size, config=config)

    def forward(self, x):
        # Forward pass through the model
        outputs = self.model(x)
        return outputs.logits