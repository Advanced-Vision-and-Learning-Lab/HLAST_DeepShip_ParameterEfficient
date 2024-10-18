#!/bin/bash

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --lr 1e-5 --patience 10 --window_length 4096 --hop_length 512 --number_mels 64 --num_epochs 150 -numBins 8 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --no-histogram --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --lr 1e-5 --patience 10 --window_length 400 --hop_length 160 --number_mels 128 --num_epochs 1 -numBins 16 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --lr 1e-5 --patience 10 --window_length 400 --hop_length 160 --number_mels 128 --num_epochs 1 -numBins 8 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa_ffn --histogram_mode parallel

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --lr 1e-5 --patience 10 --window_length 400 --hop_length 160 --number_mels 128 --num_epochs 1 -numBins 16 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --no-histogram --train_mode adapters --adapter_location mhsa_ffn --adapter_mode parallel --histogram_location None --histogram_mode None


#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --patience 10 --window_length 4096 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 --sample_rate 16000 --no-histograms_shared --adapters_shared --no-histogram --train_mode adapters --adapter_location mhsa --adapter_mode parallel --histogram_location None --histogram_mode None

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 5e-4 --patience 10 --window_length 4096 --hop_length 512 --number_mels 128 --num_epochs 150 -numBins 16 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 5e-4 --patience 10 --window_length 4096 --hop_length 512 --number_mels 128 --num_epochs 150 -numBins 16 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 5e-4 --patience 20 --window_length 4096 --hop_length 512 --number_mels 128 --num_epochs 150 -numBins 8 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel 

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 5e-4 --patience 20 --window_length 4096 --hop_length 512 --number_mels 128 --num_epochs 150 -numBins 16 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel 

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 5e-4 --patience 20 --window_length 4096 --hop_length 512 --number_mels 128 --num_epochs 150 -numBins 32 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel 

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 5e-4 --patience 20 --window_length 4096 --hop_length 512 --number_mels 128 --num_epochs 150 -numBins 4 --sample_rate 16000 --histograms_shared --no-adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location all --histogram_mode parallel 

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 5e-4 --patience 10 --window_length 4096 --hop_length 512 --number_mels 128 --num_epochs 150 -numBins 16 -RR 64 --sample_rate 16000 --no-histograms_shared --adapters_shared --no-histogram --train_mode adapters --adapter_location mhsa --adapter_mode parallel --histogram_location None --histogram_mode None

#python esc_demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 5e-4 --patience 10 --window_length 4096 --hop_length 512 --number_mels 128 --num_epochs 150 -numBins 16 -RR 64 --sample_rate 16000 --no-histograms_shared --no-adapters_shared --no-histogram --train_mode adapters --adapter_location mhsa --adapter_mode parallel --histogram_location None --histogram_mode None

