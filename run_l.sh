#!/bin/bash

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --patience 15 --window_length 1024 --hop_length 512 --number_mels 64 --num_epochs 1 -numBins 8 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --patience 15 --window_length 1024 --hop_length 512 --number_mels 64 --num_epochs 1 -numBins 8 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --patience 15 --window_length 1024 --hop_length 512 --number_mels 64 --num_epochs 150 -numBins 8 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode adapters --adapter_location mhsa --adapter_mode parallel --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --patience 15 --window_length 1024 --hop_length 512 --number_mels 64 --num_epochs 150 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode histogram --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

python demo_light.py --audio_feature LogMelFBank --train_batch_size 512 --lr 1e-3 --patience 15 --window_length 1024 --hop_length 512 --number_mels 64 --num_epochs 150 -numBins 16 -RR 64 --sample_rate 32000 --segment_length 1 --data_selection 2 --histograms_shared --adapters_shared --train_mode histogram --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

