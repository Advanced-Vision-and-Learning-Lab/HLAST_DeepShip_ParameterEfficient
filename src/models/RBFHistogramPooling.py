# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:05:26 2018
Generate histogram layer
@author: jpeeples
"""

import torch
import torch.nn as nn

class HistogramLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size,
        dim=1,
        num_bins=4,
        stride=1,
        padding=0,
        normalize_count=True,
        normalize_bins=True,
        count_include_pad=False,
        ceil_mode=False,
        output_size=None
    ):
        super(HistogramLayer, self).__init__()

        # Initialize layer properties
        self.in_channels = in_channels
        self.numBins = num_bins
        self.stride = stride
        self.kernel_size = kernel_size
        self.dim = dim
        self.padding = padding
        self.normalize_count = normalize_count
        self.normalize_bins = normalize_bins
        self.count_include_pad = count_include_pad
        self.ceil_mode = ceil_mode

        # Define convolutional layers for bin centers and widths
        self.bin_centers_conv = nn.Conv1d(self.in_channels, self.numBins, 1, bias=True)
        self.bin_widths_conv = nn.Conv1d(self.numBins, self.numBins, 1, groups=self.numBins, bias=False)

        self.hist_pool = nn.AdaptiveAvgPool1d(output_size)

        self.centers = self.bin_centers_conv.bias
        self.widths = self.bin_widths_conv.weight

    def forward(self, xx):
        # Learn bin centers
        xx = self.bin_centers_conv(xx)

        # Learn bin widths
        xx = self.bin_widths_conv(xx)

        # Apply radial basis function
        xx = torch.exp(-(xx ** 2))
        
        # Normalize bins to sum to one
        if self.normalize_bins:
            xx = self.constrain_bins(xx)

        # Apply pooling
        if self.normalize_count:
            xx = self.hist_pool(xx)
        else:
            xx = (self.kernel_size ** self.dim) * self.hist_pool(xx)

        return xx
 
    def constrain_bins(self,xx):
        #Enforce sum to one constraint across bins
        if self.dim == 1:
            n,c,l = xx.size()
            xx_sum = xx.reshape(n, c//self.numBins, self.numBins, l).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum,self.numBins,dim=1)
            xx = xx/xx_sum  
          
        else:
            raise RuntimeError('Invalid dimension for histogram layer')
         
        return xx
        
        
        
        

