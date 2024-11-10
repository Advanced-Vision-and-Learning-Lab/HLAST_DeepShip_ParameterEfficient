#!/bin/bash

python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --patience 15 --window_length 512 --hop_length 256 --number_mels 64 --num_epochs 150 -numBins 16 -RR 64 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --no-histogram --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --patience 15 --window_length 512 --hop_length 256 --number_mels 64 --num_epochs 150 -numBins 16 --sample_rate 16000 --histograms_shared --no-adapters_shared --no-histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --patience 15 --window_length 512 --hop_length 256 --number_mels 64 --num_epochs 150 -numBins 16 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --patience 15 --window_length 512 --hop_length 256 --number_mels 64 --num_epochs 150 -numBins 16 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --patience 15 --window_length 512 --hop_length 256 --number_mels 64 --num_epochs 150 -numBins 16 -RR 64 --sample_rate 16000 --no-histograms_shared --adapters_shared --no-histogram --train_mode adapters --adapter_location mhsa --adapter_mode parallel --histogram_location None --histogram_mode None

python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --patience 15 --window_length 512 --hop_length 256 --number_mels 64 --num_epochs 150 -numBins 16 -RR 64 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --no-histogram --train_mode adapters --adapter_location mhsa --adapter_mode parallel --histogram_location None --histogram_mode None


