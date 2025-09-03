#!/bin/bash

# FLS: watertank ; turntable

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --num_workers 8 --lr 1e-5 --patience 20 --num_epochs 200 -numBins 16 -RR 64 --data_selection 4 --fls_dataset turntable --histograms_shared --adapters_shared --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --num_workers 8 --lr 1e-3 --patience 20 --num_epochs 200 -numBins 16 -RR 64 --data_selection 4 --fls_dataset turntable --histograms_shared --adapters_shared --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

# Histogram (MHSA shared, 4 bins)
#python demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --num_workers 8 --lr 1e-3 --patience 20 --num_epochs 200 -numBins 4 -RR 64 --data_selection 4 --fls_dataset turntable --histograms_shared --adapters_shared --train_mode histogram --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

# Histogram (MHSA shared, 8 bins)
#python demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --num_workers 8 --lr 1e-3 --patience 20 --num_epochs 200 -numBins 8 -RR 64 --data_selection 4 --fls_dataset turntable --histograms_shared --adapters_shared --train_mode histogram --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

# Histogram (MHSA shared, 16 bins)
#python demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --num_workers 8 --lr 1e-3 --patience 20 --num_epochs 200 -numBins 16 -RR 64 --data_selection 4 --fls_dataset turntable --histograms_shared --adapters_shared --train_mode histogram --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel

# LoRA (shared, q, rank 6)
python demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --num_workers 8 --lr 1e-3 --patience 20 --num_epochs 200 -numBins 16 -RR 64 --data_selection 4 --fls_dataset turntable --histograms_shared --adapters_shared --train_mode lora --lora_target q --lora_rank 6 --lora_shared --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

# LoRA (shared, q, rank 12)
#python demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --num_workers 8 --lr 1e-3 --patience 20 --num_epochs 200 -numBins 16 -RR 64 --data_selection 4 --fls_dataset turntable --histograms_shared --adapters_shared --train_mode lora --lora_target q --lora_rank 12 --lora_shared --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

# SSF (shared, layernorm)
#python demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --num_workers 8 --lr 1e-3 --patience 20 --num_epochs 200 -numBins 16 -RR 64 --data_selection 4 --fls_dataset turntable --histograms_shared --adapters_shared --train_mode ssf --ssf_shared --ssf_mode single --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

# SSF (shared, mhsa)
#python demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --num_workers 8 --lr 1e-3 --patience 20 --num_epochs 200 -numBins 16 -RR 64 --data_selection 4 --fls_dataset turntable --histograms_shared --adapters_shared --train_mode ssf --ssf_shared --ssf_mode mhsa_only --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

# SSF (shared, mhsa+ffn)
#python demo_light.py --audio_feature LogMelFBank --train_batch_size 32 --val_batch_size 32 --test_batch_size 32 --num_workers 8 --lr 1e-3 --patience 20 --num_epochs 200 -numBins 16 -RR 64 --data_selection 4 --fls_dataset turntable --histograms_shared --adapters_shared --train_mode ssf --ssf_shared --ssf_mode full --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

