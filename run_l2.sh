#!/bin/bash

python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --patience 15 --window_length 1024 --hop_length 512 --number_mels 64 --num_epochs 150 -numBins 16 --sample_rate 16000 --histograms_shared --adapters_shared --histogram --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel



