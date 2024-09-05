#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 14:26:54 2024

@author: amir.m
"""


def extract_train_paths(file_path, output_file):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    train_paths = []
    record = False

    for line in lines:
        # Start recording after encountering 'Train indices and paths:'
        if 'Train indices and paths:' in line:
            record = True
        elif 'Validation indices and paths:' in line:
            break  # Stop recording once we reach 'Validation indices and paths:'

        # Record the lines corresponding to the training paths
        if record and line.strip() and 'Train indices and paths:' not in line:
            train_paths.append(line)

    # Write the extracted train paths to a new file
    with open(output_file, 'w') as output:
        output.write("Train indices and paths:\n")
        output.writelines(train_paths)

    print(f"Training paths extracted and saved to {output_file}")

# Specify the input and output file paths
input_file = 'all_paths.txt'  # Replace with your actual input file path
output_file = 'train_paths.txt'    # Replace with your desired output file path

# Extract and save train paths
extract_train_paths(input_file, output_file)



