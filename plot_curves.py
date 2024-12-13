import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalar_from_events(event_paths, scalar_name):
    scalar_values = []
    for event_path in event_paths:
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()
        if scalar_name in event_acc.Tags()['scalars']:
            scalar_events = event_acc.Scalars(scalar_name)
            values = [event.value for event in scalar_events]
            scalar_values.extend(values)
    return scalar_values

def list_available_scalars(event_paths):
    available_scalars = set()
    for event_path in event_paths:
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()
        available_scalars.update(event_acc.Tags()['scalars'])
    return available_scalars

def save_learning_curves(log_dir, output_path):
    # Get all event files in the log directory
    event_paths = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents.'):
                event_paths.append(os.path.join(root, file))

    # List all available scalars
    available_scalars = list_available_scalars(event_paths)
    print(f"Available Scalars in {log_dir}: {available_scalars}")
    train_loss = extract_scalar_from_events(event_paths, 'loss_epoch')
    val_loss = extract_scalar_from_events(event_paths, 'val_loss')

    if train_loss and val_loss:
        plt.figure(figsize=(6, 4))
        plt.plot(train_loss, label='Training Loss', color='blue', lw=3)
        plt.plot(val_loss, label='Validation Loss', color='orange', lw=3)
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Loss', fontsize=15)
        plt.title('Learning Curves', fontsize=18)
        plt.legend(loc="best", fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        print(f"Required scalars ('loss_epoch', 'val_loss') not found in {log_dir}.")

def save_accuracy_curves(log_dir, output_path):
    # Get all event files in the log directory
    event_paths = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents.'):
                event_paths.append(os.path.join(root, file))

    # List all available scalars
    available_scalars = list_available_scalars(event_paths)
    print(f"Available Scalars in {log_dir}: {available_scalars}")
    train_acc = extract_scalar_from_events(event_paths, 'train_acc')
    val_acc = extract_scalar_from_events(event_paths, 'val_acc')

    # Plot accuracy curves if both scalars are available
    if train_acc and val_acc:
        plt.figure(figsize=(6, 4))
        plt.plot(train_acc, label='Training Accuracy', color='green', lw=3)
        plt.plot(val_acc, label='Validation Accuracy', color='red', lw=3)
        plt.xlabel('Epochs', fontsize=15)
        plt.ylabel('Accuracy', fontsize=15)
        plt.title('Accuracy Curves', fontsize=18)
        plt.legend(loc="best", fontsize=12)
        plt.grid(True)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)

        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        print(f"Required scalars ('train_acc', 'val_acc') not found in {log_dir}.")

def process_all_logs(tb_logs_base_dir='tb_logs', features_base_dir='features/Curves'):
    torch.set_float32_matmul_precision('medium')

    # Walk through each experiment in tb_logs
    for experiment in os.listdir(tb_logs_base_dir):
        exp_path = os.path.join(tb_logs_base_dir, experiment)
        if not os.path.isdir(exp_path):
            continue  # Skip if not a directory

        # Walk through each run in the experiment
        for run in os.listdir(exp_path):
            run_path = os.path.join(exp_path, run)
            if not os.path.isdir(run_path):
                continue  # Skip if not a directory

            metrics_path = os.path.join(run_path, 'metrics')
            if not os.path.isdir(metrics_path):
                print(f"Metrics directory not found for {run_path}. Skipping.")
                continue

            # Define output directory with similar structure
            output_dir = os.path.join(features_base_dir, experiment, run)
            os.makedirs(output_dir, exist_ok=True)

            # Save learning curves
            learning_curves_output_path = os.path.join(output_dir, "learning_curves.png")
            save_learning_curves(metrics_path, learning_curves_output_path)

            # Save accuracy curves
            accuracy_curves_output_path = os.path.join(output_dir, "accuracy_curves.png")
            save_accuracy_curves(metrics_path, accuracy_curves_output_path)

def main():
    process_all_logs()

if __name__ == "__main__":
    main()
