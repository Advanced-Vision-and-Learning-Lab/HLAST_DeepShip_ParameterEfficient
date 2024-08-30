#!/bin/bash

python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 8 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location all --histogram_mode parallel


#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 4 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 8 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 64 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 4 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location ffn --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 8 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location ffn --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 64 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location ffn --histogram_mode parallel

