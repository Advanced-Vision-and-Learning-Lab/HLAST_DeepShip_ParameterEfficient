# -*- coding: utf-8 -*-
"""
Parameters for histogram layer experiments
Only change parameters in this file before running
demo.py
@author: jpeeples 
"""
import os
import sys

def Parameters(args):
    ######## ONLY CHANGE PARAMETERS BELOW ######## 
    #optimizer selection
    optimizer = args.optimizer
    
    #Flag to use histogram model or baseline global average pooling (GAP)
    # Set to True to use histogram layer and False to use GAP model
    histograms_shared = args.histograms_shared
    adapters_shared = args.adapters_shared
    
    #Select dataset. Set to number of desired dataset
    data_selection = args.data_selection
    Dataset_names = {0: 'DeepShip', 1: 'ShipsEar', 2: 'VTUAD'}
    
    #Number of bins for histogram layer. Recommended values are 4, 8 and 16.
    #Set number of bins to powers of 2 (e.g., 2, 4, 8, etc.)
    #For HistRes_B models using ResNet18 and ResNet50, do not set number of bins
    #higher than 128 and 512 respectively. Note: a 1x1xK convolution is used to
    #downsample feature maps before binning process. If the bin values are set
    #higher than 128 or 512 for HistRes_B models using ResNet18 or ResNet50
    #respectively, than an error will occur due to attempting to reduce the number of
    #features maps to values less than one
    numBins = args.numBins
    RR = args.RR
    #Flag for feature extraction. False, train whole model. True, only update
    #fully connected and histogram layers parameters (default: False)
    #Flag to use pretrained model from ImageNet or train from scratch (default: True)
    #Flag to add BN to convolutional features (default:True)
    #Location/Scale at which to apply histogram layer (default: 5 (at the end))
    train_mode = args.train_mode
    use_pretrained = args.use_pretrained

    #Set learning rate for new layers
    #Recommended values are .01 (used in paper) or .001
    lr = args.lr
    
    #Set whether to have the histogram layer inline or parallel (default: parallel)
    #Set whether to use sum (unnormalized count) or average pooling (normalized count)
    # (default: average pooling)
    #Set whether to enforce sum to one constraint across bins (default: True)
    parallel = True
    normalize_count = True
    normalize_bins = True
    
    #Batch size for training and epochs. If running experiments on single GPU (e.g., 2080ti),
    #training batch size is recommended to be 64. If using at least two GPUs,
    #the recommended training batch size is 128 (as done in paper)
    #May need to reduce batch size if CUDA out of memory issue occurs
    batch_size = {'train': args.train_batch_size, 'val': args.val_batch_size, 'test': args.test_batch_size} 
    num_epochs = args.num_epochs
    
    #Patience is the number of epochs to observe if a metric (loss or accuarcy)
    #is minimized or maximized
    patience = args.patience
    window_length = args.window_length
    hop_length = args.hop_length
    number_mels = args.number_mels
    
    #Pin memory for dataloader (set to True for experiments)
    pin_memory = False
    
    #Set number of workers, i.e., how many subprocesses to use for data loading.
    #Usually set to 0 or 1. Can set to more if multiple machines are used.
    #Number of workers for experiments for two GPUs was three
    num_workers = 8
    
    #Select audio feature for DeepShip 
    feature = args.audio_feature
    
    #Set to True if more than one GPU was used
    Parallelize_model = True

    segment_length = args.segment_length

    #sample_rate ={'DeepShip': 32000}
    sample_rate = args.sample_rate

    #ResNet models to use for each dataset
    Model_name = args.model
    
    #Number of classes in each dataset
    # num_classes = {'DeepShip': 4}
    
    # #Number of runs and/or splits for each dataset
    # Splits = {'DeepShip': 3}
    
    # Dataset = Dataset_names[data_selection]
    # data_dir = Data_dirs[Dataset]

    
    new_dir_p = './Datasets/DeepShip/'
    new_dir = '{}Segments_{}s_{}hz/'.format(new_dir_p,segment_length,sample_rate)
    
    #Return dictionary of parameters
    Params = {'histograms_shared': histograms_shared,'adapters_shared': adapters_shared,
                          'sample_rate':sample_rate,'segment_length':segment_length,'new_dir':new_dir,
                          'optimizer': optimizer,'num_workers': num_workers,'lr': lr,'batch_size' : batch_size, 
                          'num_epochs': num_epochs,'normalize_count': normalize_count, 'data_selection':data_selection,
                          'normalize_bins': normalize_bins,'parallel': parallel,
                          'numBins': numBins,'RR': RR,'Model_name': Model_name, 
                          'train_mode': train_mode, 'use_pretrained': use_pretrained,
                          'pin_memory': pin_memory,'Parallelize': Parallelize_model,
                          'feature': feature, 'patience': patience,
                          'window_length':window_length,'hop_length':hop_length,'number_mels':number_mels,
                          'adapter_location': args.adapter_location,'adapter_mode': args.adapter_mode,
                          'histogram_location': args.histogram_location,'histogram_mode': args.histogram_mode}
    return Params

