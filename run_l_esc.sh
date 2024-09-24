#!/bin/bash

python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --patience 10 --window_length 4096 --hop_length 1024 --number_mels 64 --num_epochs 150 -numBins 8 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --no-histogram --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --patience 10 --window_length 4096 --hop_length 512 --number_mels 128 --num_epochs 150 -numBins 8 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --no-histogram --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --patience 10 --window_length 4096 --hop_length 512 --number_mels 128 --num_epochs 150 -numBins 8 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --no-histogram --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --lr 1e-5 --patience 10 --window_length 400 --hop_length 160 --number_mels 128 --num_epochs 150 -numBins 8 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --no-histogram --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None



