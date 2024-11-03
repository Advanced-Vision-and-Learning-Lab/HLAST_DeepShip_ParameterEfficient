#!/bin/bash

python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --patience 25 --window_length 8192 --hop_length 1024 --number_mels 128 --num_epochs 100 -numBins 8 --sample_rate 16000 --histograms_shared --adapters_shared --no-histogram --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None


#python demo_light.py --audio_feature MelSpec --train_batch_size 4 --lr 1e-5 --patience 15 --window_length 8192 --hop_length 1024 --number_mels 1024 --num_epochs 150 -numBins 16 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --no-histogram --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None



python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 5e-4 --patience 25 --window_length 8192 --hop_length 1024 --number_mels 128 --num_epochs 100 -numBins 8 --sample_rate 16000 --histograms_shared --adapters_shared --no-histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None


python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 5e-4 --patience 25 --window_length 8192 --hop_length 1024 --number_mels 128 --num_epochs 100 -numBins 8 --sample_rate 16000 --histograms_shared --adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel


python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 5e-4 --patience 25 --window_length 8192 --hop_length 1024 --number_mels 128 --num_epochs 100 -numBins 8 -RR 256 --sample_rate 16000 --histograms_shared --adapters_shared --histogram --train_mode adapters --adapter_location mhsa --adapter_mode parallel --histogram_location None --histogram_mode None
