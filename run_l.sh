#!/bin/bash


python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode lora --lora_target q --lora_rank 6 --lora_shared


python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode ssf --ssf_shared --ssf_mode full

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 2 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode ssf --ssf_shared --ssf_mode mhsa_only

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 2 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode ssf --no-ssf_shared --ssf_mode full

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 2 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode ssf --no-ssf_shared --ssf_mode mhsa_only






#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 200 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode lora --lora_target q --lora_rank 6 --lora_shared


#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 --sample_rate 32000 --segment_length 1 --data_selection 2 --histograms_shared --adapters_shared --train_mode bias --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None




#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 --sample_rate 16000 --segment_length 5 --data_selection 0 --histograms_shared --adapters_shared --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 --sample_rate 16000 --segment_length 5 --data_selection 0 --histograms_shared --adapters_shared --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 0 --histograms_shared --adapters_shared --train_mode adapters --adapter_location mhsa --adapter_mode parallel --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 200 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 0 --histograms_shared --adapters_shared --train_mode histogram --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel




#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode adapters --adapter_location mhsa --adapter_mode parallel --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 -RR 64 --sample_rate 16000 --segment_length 5 --data_selection 1 --histograms_shared --adapters_shared --train_mode histogram --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel





#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 --sample_rate 32000 --segment_length 1 --data_selection 2 --histograms_shared --adapters_shared --train_mode linear_probing --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-5 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 --sample_rate 32000 --segment_length 1 --data_selection 2 --histograms_shared --adapters_shared --train_mode full_fine_tune --adapter_location None --adapter_mode None --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 --sample_rate 32000 --segment_length 1 --data_selection 2 --histograms_shared --adapters_shared --train_mode adapters --adapter_location mhsa --adapter_mode parallel --histogram_location None --histogram_mode None

#python demo_light.py --audio_feature LogMelFBank --train_batch_size 64 --lr 1e-3 --num_worker 8 --patience 20 --window_length 2048 --hop_length 512 --number_mels 128 --num_epochs 1 -numBins 16 --sample_rate 32000 --segment_length 1 --data_selection 2 --histograms_shared --adapters_shared --train_mode histogram --adapter_location None --adapter_mode None --histogram_location mhsa --histogram_mode parallel






