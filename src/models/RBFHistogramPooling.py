# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 12:05:26 2018
Generate histogram layer
@author: jpeeples
"""

import torch
import torch.nn as nn
import numpy as np
import pdb
from scipy.special import gamma
import torch.nn.functional as F
class HistogramLayer(nn.Module):
    def __init__(self,in_channels,kernel_size,dim=1,num_bins=4,
                  stride=1,padding=0,normalize_count=True,normalize_bins = True,
                  count_include_pad=False, ceil_mode=False, df=3):

        # inherit nn.module
        super(HistogramLayer, self).__init__()

        # define layer properties
        # histogram bin data
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
        
        self.df = df  # Degrees of freedom for t-distribution
        # Initialize df as a learnable parameter
        #self.df = nn.Parameter(torch.tensor(float(df)), requires_grad=True)

        #For each data type, apply two 1x1 convolutions, 1) to learn bin center (bias)
        # and 2) to learn bin width
        # Time series/ signal Data
        if self.dim == 1:
            #pdb.set_trace()
            self.bin_centers_conv = nn.Conv1d(self.in_channels,self.numBins,1,
                                            groups=1,bias=True)
            
            
            #self.activation = nn.ReLU()  # or 
            #self.activation = nn.GELU()
            
            
            self.bin_widths_conv = nn.Conv1d(self.numBins,
                                              self.numBins,1,
                                              groups=self.numBins,
                                              bias=False)


            self.hist_pool = nn.AvgPool1d(self.kernel_size,stride=self.stride,
                                              padding=self.padding,ceil_mode=self.ceil_mode,
                                              count_include_pad=self.count_include_pad)
            
            self.centers = self.bin_centers_conv.bias
            self.widths = self.bin_widths_conv.weight
            
            
            
            # Zero initialization
            #nn.init.zeros_(self.bin_centers_conv.weight)
            #nn.init.zeros_(self.bin_centers_conv.bias)
            #nn.init.zeros_(self.bin_widths_conv.weight)

    
            # Calculate t-distribution normalization factor
            #self.t_norm_factor = gamma((self.df + 1) / 2) / (np.sqrt(self.df * np.pi) * gamma(self.df / 2))
            #self.t_norm_factor = torch.tensor(self.t_norm_factor, dtype=torch.float32)


            # Initialize bins equally spaced
            #self.initialize_bins()
        else:
            raise RuntimeError('Invalid dimension for histogram layer')

    # def initialize_bins(self):
    #     # Estimate data range (you may need to adjust this based on your data)
    #     min_val, max_val = -1.0, 1.0  # Assuming normalized input data

    #     # Initialize bin centers
    #     bin_centers = torch.linspace(min_val, max_val, self.numBins)
    #     self.bin_centers_conv.bias.data = bin_centers

    #     # Initialize bin widths
    #     bin_width = (max_val - min_val) / (self.numBins - 1)
    #     self.bin_widths_conv.weight.data.fill_(1.0 / bin_width)
        
        
        
    def forward(self,xx):        
        
        #pdb.set_trace()
        #Pass through first convolution to learn bin centers
        xx = self.bin_centers_conv(xx)
        
        #xx = self.activation(xx)
        
        #Pass through second convolution to learn bin widths
        xx = self.bin_widths_conv(xx)
        
        #xx = self.activation(xx)
        
        # Apply t-distribution
        #xx = (1 + (1/self.df) * xx**2)**(-(self.df+1)/2)

        #Pass through radial basis function
        #xx = torch.exp(-nn.functional.gelu(xx)**2)
        xx = torch.exp(-(xx**2))

        #Enforce sum to one constraint
        # Add small positive constant in case sum is zero
        if(self.normalize_bins):
            xx = self.constrain_bins(xx)
        
        #Get localized histogram output, if normalize, average count
        if(self.normalize_count):
            xx = self.hist_pool(xx)

        else:
            xx = np.prod(np.asarray(self.hist_pool.kernel_size))*self.hist_pool(xx)
     
        return xx
        
 

    # def forward(self, xx):


        
    #     # Compute global feature vector using global average pooling
    #     global_feature = F.adaptive_avg_pool1d(xx, 1).view(xx.size(0), xx.size(1))

    #     # Normalize both local and global features for cosine similarity calculation
    #     xx_norm = F.normalize(xx, p=2, dim=1)
    #     global_feature_norm = F.normalize(global_feature, p=2, dim=1)

    #     # Compute cosine similarity for each position
    #     cosine_similarity = torch.bmm(xx_norm.permute(0, 2, 1), global_feature_norm.unsqueeze(2)).squeeze(2)

    #     # Pass through first convolution to learn bin centers
    #     xx_centers = self.bin_centers_conv(xx)

    #     # Pass through second convolution to learn bin widths
    #     xx_widths = self.bin_widths_conv(xx_centers)

    #     # Apply cosine similarity as a weight to modify the binning process
    #     xx_weighted = xx_widths * cosine_similarity.unsqueeze(1)

    #     # Apply radial basis function (RBF) or Gaussian-like transformation
    #     xx_rbf = torch.exp(-(xx_weighted**2))

    #     #Enforce sum to one constraint
    #     # Add small positive constant in case sum is zero
    #     if(self.normalize_bins):
    #         xx_rbf = self.constrain_bins(xx_rbf)
        
    #     #Get localized histogram output, if normalize, average count
    #     if(self.normalize_count):
    #         xx_rbf = self.hist_pool(xx_rbf)

    #     else:
    #         xx_rbf = np.prod(np.asarray(self.hist_pool.kernel_size))*self.hist_pool(xx_rbf)
     
    #     return xx_rbf 

    
    
 
    def constrain_bins(self,xx):
        #Enforce sum to one constraint across bins
        # Time series/ signal Data
        if self.dim == 1:
            n,c,l = xx.size()
            xx_sum = xx.reshape(n, c//self.numBins, self.numBins, l).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum,self.numBins,dim=1)
            xx = xx/xx_sum  
        
        # Image Data
        elif self.dim == 2:
            n,c,h,w = xx.size()
            xx_sum = xx.reshape(n, c//self.numBins, self.numBins, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum,self.numBins,dim=1)
            xx = xx/xx_sum  
        
        # Spatial/Temporal or Volumetric Data
        elif self.dim == 3:
            n,c,d,h,w = xx.size()
            xx_sum = xx.reshape(n, c//self.numBins, self.numBins,d, h, w).sum(2) + torch.tensor(10e-6)
            xx_sum = torch.repeat_interleave(xx_sum,self.numBins,dim=1)
            xx = xx/xx_sum   
            
        else:
            raise RuntimeError('Invalid dimension for histogram layer')
         
        return xx
        
        
        
        

