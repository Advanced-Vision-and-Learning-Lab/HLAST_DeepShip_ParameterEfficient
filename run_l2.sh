#!/bin/bash

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --spec_norm True --lr 1e-5 --patience 25 --num_epochs 150 -numBins 16 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --no-histogram --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 8 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 16 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location ffn --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 15 --num_epochs 1 -numBins 8 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa_ffn --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 16 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa_ffn --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 4 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa_ffn --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 8 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa_ffn --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 64 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa_ffn --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 15 --num_epochs 1 -numBins 32 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa_ffn --histogram_mode parallel

#python demo_light.py --audio_feature STFT --train_batch_size 64 --lr 1e-5 --patience 25 --num_epochs 150 -numBins 16 --sample_rate 16000 --no-histograms_shared --adapters_shared --no-histogram --train_mode adapters --adapter_location mhsa_ffn --adapter_mode parallel --histogram_location None --histogram_mode None


