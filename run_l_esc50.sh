#!/bin/bash

# HPT
python demo_ESC50.py --train_mode histogram --audio_feature LogMelFBank --train_batch_size 64 --val_batch_size 64 --num_workers 8 --lr 1e-3 --numBins 8 --histogram_location mhsa --histogram_mode parallel --histograms_shared --window_length 400 --hop_length 160 --number_mels 64

# Full fine-tune
python demo_ESC50.py --train_mode full_fine_tune --train_batch_size 64 --val_batch_size 64 --num_workers 8 --lr 1e-5 --window_length 400 --hop_length 160 --number_mels 64

# Linear probe 
python demo_ESC50.py --train_mode linear_probing --train_batch_size 64 --val_batch_size 64 --num_workers 8 --window_length 400 --hop_length 160 --number_mels 64

# Adapters 
python demo_ESC50.py --train_mode adapters --train_batch_size 64 --val_batch_size 64 --adapter_location mhsa --adapter_mode parallel --adapters_shared --RR 128 --num_workers 8 --window_length 400 --hop_length 160 --number_mels 64

# LoRA 
python demo_ESC50.py --train_mode lora --train_batch_size 64 --val_batch_size 64 --lora_target q --lora_rank 6 --lora_shared --window_length 400 --hop_length 160 --number_mels 64 --lr 1e-3 --num_workers 8 --window_length 400 --hop_length 160 --number_mels 64

# SSF on MHSA only, shared
python demo_ESC50.py --train_mode ssf --train_batch_size 64 --val_batch_size 64 --ssf_mode mhsa_only --ssf_shared --window_length 400 --hop_length 160 --number_mels 64 --lr 1e-3 --num_workers 8 --window_length 400 --hop_length 160 --number_mels 64

