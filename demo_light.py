import pdb
import numpy as np
import argparse
import torch
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from Demo_Parameters import Parameters
from Utils.LitModel import LitModel

from Datasets.Get_preprocessed_data import process_data
from Datasets.SSDataModule import SSAudioDataModule

from Datasets.ShipsEar_Data_Preprocessing import Generate_Segments
from Datasets.ShipsEar_dataloader import ShipsEarDataModule

import os
import zipfile
from Datasets.Create_Combined_VTUAD import Create_Combined_VTUAD
from Datasets.VTUAD_DataModule import AudioDataModule

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def unzip_if_needed(zip_path, extract_to):
    """
    Check if the zip file is already unzipped. If not, unzip it.
    """
    if not os.path.exists(extract_to):
        print(f"Unzipping {zip_path} to {extract_to}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction of {zip_path} completed.")
    else:
        print(f"{extract_to} already exists. Skipping extraction.")
        
def main(Params):

    model_name = Params['Model_name']

    numBins = Params['numBins']
    RR = Params['RR']
    h_shared = Params['histograms_shared']
    a_shared = Params['adapters_shared']
    
    batch_size = Params['batch_size']
    t_batch_size = batch_size['train']

    num_workers = Params['num_workers']
    
    run_number = 0
    seed_everything(run_number+1, workers=True)
    new_dir = Params["new_dir"] 
    
    if Params['data_selection'] == 0:
        process_data(sample_rate=Params['sample_rate'], segment_length=Params['segment_length'])
        data_module = SSAudioDataModule(data_dir=new_dir, batch_size=batch_size, num_workers=num_workers)
        data_module.prepare_data()
        num_classes = 4
        
    elif Params['data_selection'] == 1:
        dataset_dir = './Datasets/ShipsEar/'
        Generate_Segments(dataset_dir, target_sr=16000, segment_length=5)
        data_module = ShipsEarDataModule(parent_folder='./Datasets/ShipsEar', batch_size=batch_size, num_workers=num_workers)
        num_classes = 5
        
    elif Params['data_selection'] == 2:
        base_dir = './Datasets/VTUAD'
        
        scenarios = [
            'inclusion_2000_exclusion_4000',
            'inclusion_3000_exclusion_5000',
            'inclusion_4000_exclusion_6000'
        ]
        zip_files = [
            'inclusion_2000_exclusion_4000.zip',
            'inclusion_3000_exclusion_5000.zip',
            'inclusion_4000_exclusion_6000.zip'
        ]
        
        # Check and unzip all zip files
        for zip_file, scenario in zip(zip_files, scenarios):
            zip_path = os.path.join(base_dir, zip_file)
            extract_to = os.path.join(base_dir, scenario)
            unzip_if_needed(zip_path, extract_to)
        
        # Check and create combined scenario
        combined_scenario = 'combined_scenario'
        combined_path = os.path.join(base_dir, combined_scenario)
        
        if not os.path.exists(combined_path):
            print(f"Creating combined scenario at {combined_path}...")
            create_combined = Create_Combined_VTUAD(
                base_dir=base_dir,
                scenarios=scenarios,
                combined_scenario=combined_scenario
            )
            create_combined.create_combined_scenario()
            print(f"Combined scenario '{combined_scenario}' created successfully.")
        else:
            print(f"Combined scenario '{combined_scenario}' already exists. Skipping creation.")
        
        # Choose scenario
        available_scenarios = scenarios + [combined_scenario]
        print("Available scenarios:")
        for idx, scenario in enumerate(available_scenarios, 1):
            print(f"{idx}. {scenario}")
        
        chosen_scenario = combined_scenario # chosen_scenario = 'inclusion_2000_exclusion_4000'
              
        data_module = AudioDataModule(base_dir=base_dir, scenario_name=chosen_scenario,
                                      batch_size=batch_size, num_workers=num_workers)
        num_classes = 5
        
    else:
        raise ValueError('Invalid data selection: must be 0, 1, or 2')
    
    DataName = "DeepShip" if Params['data_selection'] == 0 else \
             "ShipsEar" if Params['data_selection'] == 1 else \
             "VTUAD" if Params['data_selection'] == 2 else \
             "Invalid selection"
    print('\nStarting Experiments for ' + DataName)
    
    numRuns = 1
    progress_bar=False
    
    torch.set_float32_matmul_precision('medium')
    all_val_accs = []
    all_test_accs = []
    
    for run_number in range(0, numRuns):
        
        if run_number != 0:
            seed_everything(run_number+1, workers=True)
                 
        print(f'\nStarting Run {run_number}\n')
    
        checkpoint_callback = ModelCheckpoint(
            monitor='val_acc',
            filename='best-{epoch:02d}-{val_acc:.2f}',
            save_top_k=1,
            mode='max',
            verbose=True,
            save_weights_only=True
        )
    
        early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=Params['patience'],
            verbose=True,
            mode='min'
        )

        model_AST = LitModel(Params, model_name, num_classes, numBins, RR)

        num_params = count_trainable_params(model_AST)
        print(f'Total Trainable Parameters: {num_params}')

        logger = TensorBoardLogger(
            save_dir = (
                f"tb_logs/{DataName}_{Params['feature']}_b{t_batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
                f"_AdaptShared{a_shared}_RR{RR}_{Params['adapter_location']}_{Params['adapter_mode']}_Shared{h_shared}"
                f"_{numBins}bins_{Params['histogram_location']}_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}"
                f"_m{Params['number_mels']}_ssf{Params['ssf_mode']}_sh{Params['ssf_shared']}_lora{Params['lora_target']}_R{Params['lora_rank']}_Share{Params['lora_shared']}_bias{Params['bias_mode']}/Run_{run_number}"
            ),
            name="metrics"
        )



        trainer = L.Trainer(
            max_epochs=Params['num_epochs'],
            callbacks=[early_stopping_callback, checkpoint_callback],
            deterministic=False,
            logger=logger,
            log_every_n_steps=20,
            enable_progress_bar=progress_bar,
            accelerator='gpu',       
        	devices="auto"
        )
        
        trainer.fit(model=model_AST, datamodule=data_module) 
        
        best_val_acc = checkpoint_callback.best_model_score.item()
        all_val_accs.append(best_val_acc)
    
        best_model_path = checkpoint_callback.best_model_path
        best_model = LitModel.load_from_checkpoint(
            checkpoint_path=best_model_path,
            Params=Params,
            model_name=model_name,
            num_classes=num_classes,
            pretrained_loaded=True,
            run_number=run_number
        )
    
        test_results = trainer.test(model=best_model, datamodule=data_module)
        
        best_test_acc = test_results[0]['test_acc']
        all_test_accs.append(best_test_acc)
    
        results_filename = (
            f"tb_logs/{DataName}_{Params['feature']}_b{t_batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
            f"_AdaptShared{a_shared}_RR{RR}_{Params['adapter_location']}_{Params['adapter_mode']}_Shared{h_shared}"
            f"_{numBins}bins_{Params['histogram_location']}_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}"
            f"_m{Params['number_mels']}_ssf{Params['ssf_mode']}_sh{Params['ssf_shared']}_lora{Params['lora_target']}_R{Params['lora_rank']}_Share{Params['lora_shared']}_bias{Params['bias_mode']}/Run_{run_number}/metrics.txt"
        )

        with open(results_filename, "a") as file:
            file.write(f"Run_{run_number}:\n\n")
            file.write(f"Best Validation Accuracy: {best_val_acc:.4f}\n")
            file.write(f"Best Test Accuracy: {best_test_acc:.4f}\n\n")
    
    overall_avg_val_acc = np.mean(all_val_accs)
    overall_std_val_acc = np.std(all_val_accs)
    
    overall_avg_test_acc = np.mean(all_test_accs)
    overall_std_test_acc = np.std(all_test_accs)
    
    summary_filename = (
            f"tb_logs/{DataName}_{Params['feature']}_b{t_batch_size}_{Params['sample_rate']}_{Params['train_mode']}"
            f"_AdaptShared{a_shared}_RR{RR}_{Params['adapter_location']}_{Params['adapter_mode']}_Shared{h_shared}"
            f"_{numBins}bins_{Params['histogram_location']}_{Params['histogram_mode']}_w{Params['window_length']}_h{Params['hop_length']}"
            f"_m{Params['number_mels']}_ssf{Params['ssf_mode']}_sh{Params['ssf_shared']}_lora{Params['lora_target']}_R{Params['lora_rank']}_Share{Params['lora_shared']}_bias{Params['bias_mode']}/summary_metrics.txt"
        )

    with open(summary_filename, "a") as file:
        file.write("Overall Results Across All Runs\n\n")
        file.write(f"Overall Average of Best Validation Accuracies: {overall_avg_val_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Validation Accuracies: {overall_std_val_acc:.4f}\n\n")
        file.write(f"Overall Average of Best Test Accuracies: {overall_avg_test_acc:.4f}\n")
        file.write(f"Overall Standard Deviation of Best Test Accuracies: {overall_std_test_acc:.4f}\n\n")

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run histogram experiments')
    parser.add_argument('--model', type=str, default='AST',
                        help='Select baseline model architecture')
    parser.add_argument('--histograms_shared', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use histogram shared')
    parser.add_argument('--adapters_shared', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use adapter shared')
    parser.add_argument('--data_selection', type=int, default=1,
                        help='Dataset selection: See Demo_Parameters for full list of datasets')
    parser.add_argument('-numBins', type=int, default=16,
                        help='Number of bins for histogram layer. Recommended values are 4, 8 and 16. (default: 16)')
    parser.add_argument('-RR', type=int, default=128,
                        help='Adapter Reduction Rate (default: 128)')
    parser.add_argument('--train_mode', type=str, default='ssf',
                        help='full_fine_tune or linear_probing or adapters or histogram or ssf')
    parser.add_argument('--use_pretrained', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use pretrained model or train from scratch (default: True)')
    parser.add_argument('--train_batch_size', type=int, default=16,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--val_batch_size', type=int, default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test_batch_size', type=int, default=32,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=1,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--audio_feature', type=str, default='LogMelFBank',
                        help='Audio feature for extraction')
    parser.add_argument('--patience', type=int, default=25,
                        help='Number of epochs to train each model for (default: 50)')
    parser.add_argument('--window_length', type=int, default=2048,
                        help='window length')
    parser.add_argument('--hop_length', type=int, default=512,
                        help='hop length')
    parser.add_argument('--number_mels', type=int, default=128,
                        help='number of mels')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Dataset Sample Rate'),
    parser.add_argument('--segment_length', type=int, default=5,
                        help='Dataset Segment Length'),
    parser.add_argument('--adapter_location', type=str, default='None',
                        help='Location for the adapter layers (default: ffn)')
    parser.add_argument('--adapter_mode', type=str, default='None',
                        help='Mode for the adapter layers (default: parallel)')
    parser.add_argument('--histogram_location', type=str, default='None',
                        help='Location for the histogram layers (default: ffn)')
    parser.add_argument('--histogram_mode', type=str, default='None',
                        help='Mode for the histogram layers (default: parallel)')
    parser.add_argument('--lora_target', type=str, default='q',
                        help='location for the lora (default: q)')
    parser.add_argument('--lora_rank', type=int, default=6,
                        help='rank for the lora (default: 4)')
    parser.add_argument('--lora_shared', default=False, action=argparse.BooleanOptionalAction,
                        help='Flag to use lora shared')
    parser.add_argument('--bias_mode', type=str, default='full',
                        help='bias selection (default: full)')   
    parser.add_argument('--ssf_shared', default=True, action=argparse.BooleanOptionalAction,
                        help='Flag to use ssf shared')
    parser.add_argument('--ssf_mode', type=str, default='full',
                        help='ssf_mode selection (default: full)')                  
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    params = Parameters(args)
    main(params)
