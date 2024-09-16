import matplotlib.pyplot as plt
import numpy as np
import pdb

def read_cosine_similarity(filename):
    """
    Read the cosine similarity values from the saved text file.
    """
    cosine_similarities = {}
    model_names = []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("Layer"):
                # Parse the layer index
                layer_index = int(line.split()[1])
                # Parse the similarity values and corresponding model names
                values = line.split('[')[1].split(']')[0].split(',')
                cosine_similarities[layer_index] = []
                for value in values:
                    model_name, similarity = value.strip().split(': ')
                    cosine_similarities[layer_index].append(float(similarity))
                    if model_name not in model_names:
                        model_names.append(model_name)
    
    return cosine_similarities, model_names

def plot_cosine_similarity(cosine_similarities, model_names, output_path='cosine_similarity_plot.png'):
    """
    Plot the cosine similarity values for each layer across models (excluding full_fine_tune) and save the plot.
    """
    num_layers = len(cosine_similarities)
    layers = np.arange(num_layers)

    plt.figure(figsize=(8, 6))  # Larger figure size for better resolution
    
    # Plot each model's similarity values, skipping 'full_fine_tune'
    for i, model_name in enumerate(model_names):
        if model_name != 'full_fine_tune':
            similarities = [cosine_similarities[layer][i] for layer in layers]
            plt.plot(layers, similarities, label=model_name, marker='o')
    
    # Set plot titles and labels
    plt.title('Cosine Similarity for Each Layer Across Models', fontsize=20)
    plt.xlabel('Layer Number', fontsize=18)
    plt.ylabel('Cosine Similarity', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    
    # Customize tick label sizes
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Save the plot with high resolution
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    
# Read cosine similarity data from the saved text file
cosine_similarities, model_names = read_cosine_similarity('features/logmel_cosine_similarity_results.txt')

# Generate and save the plot
plot_cosine_similarity(cosine_similarities, model_names, output_path='features/logmel_cosine_similarity_results.png')

flattened_values = [value for sublist in cosine_similarities.values() for value in sublist[1:]]
cosine_similarities_reshaped = np.array(flattened_values).reshape(12, 3)



def compute_layer_wise_mad_and_std(z_scores):
    """ 
    Compute Mean Absolute Deviation (MAD) and standard deviation of Z-Scores from 1 for each layer.
    The MAD and standard deviation are calculated layer-wise and then averaged across all layers for each model.
    """
    # Subtract 1 from each z-score
    deviation_from_1 = np.abs(z_scores - 1)
    
    # Compute the mean absolute deviation and standard deviation across layers for each model
    mad = np.mean(deviation_from_1, axis=0)
    std = np.std(deviation_from_1, axis=0)
    
    return mad, std

mad_results, std_results = compute_layer_wise_mad_and_std(cosine_similarities_reshaped)

# Print results with both MAD and Standard Deviation
model_names = ['Linear Probing', 'Histogram', 'Adapters']
for i, model_name in enumerate(model_names):
    print(f"{model_name} MAD compared to Full Fine-Tune: {mad_results[i]:.4f}, STD: {std_results[i]:.4f}\n")

# Save the results to a file
with open('features/mad_std_layer_comparison_cosine_logmel.txt', 'w') as file:
    for i, model_name in enumerate(model_names):
        file.write(f"{model_name} MAD compared to Full Fine-Tune: {mad_results[i]:.4f}, STD: {std_results[i]:.4f}\n")
        
        
        
        
        
        
        
        
        
        
def read_cosine_similarity(filename):
    cosine_similarities = {}
    model_names = []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("Layer"):
                layer = int(line.split()[1])
                cosine_similarities[layer] = {}
            elif ":" in line:
                model, stats = line.strip().split(":")
                model = model.strip()
                mean, std = stats.split(",")
                mean = float(mean.split("=")[1].strip())
                std = float(std.split("=")[1].strip())
                cosine_similarities[layer][model] = {'mean': mean, 'std': std}
                if model not in model_names:
                    model_names.append(model)
    
    return cosine_similarities, model_names        
        
def plot_cosine_similarity_with_std(cosine_similarities, model_names):
    layers = list(cosine_similarities.keys())
    
    plt.figure(figsize=(8, 6))
    
    for model in model_names:
        means = [cosine_similarities[layer][model]['mean'] for layer in layers]
        stds = [cosine_similarities[layer][model]['std'] for layer in layers]
        
        plt.errorbar(layers, means, yerr=stds, fmt='-o', capsize=5, label=model)
    
    plt.title("Cosine Similarity (Mean ± Std)", fontsize=16)
    plt.xlabel("Layer", fontsize=14)
    plt.ylabel("Cosine Similarity", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.xticks(layers)
    plt.savefig("features/mean_std_logmel_cosine_similarity.png", dpi=300, bbox_inches='tight')
    plt.close()

# Read the cosine similarity results
filename = 'features/mean_std_logmel_cosine_similarity_results.txt'
cosine_similarities, model_names = read_cosine_similarity(filename)

# Create visualization
plot_cosine_similarity_with_std(cosine_similarities, model_names)        
        
        
        
        
        


def read_kruskal_wallis_results(filename):
    """
    Read the Kruskal-Wallis test results from the saved text file.
    """
    kruskal_results = {}
    model_names = []
    
    with open(filename, 'r') as file:
        current_layer = None
        for line in file:
            if line.startswith("Layer"):
                current_layer = int(line.split()[1])
                kruskal_results[current_layer] = {}
            elif "vs full_fine_tune:" in line:
                model_name = line.split("vs")[0].strip()
                if model_name not in model_names:
                    model_names.append(model_name)
            elif "Statistic:" in line:
                statistic = float(line.split(":")[1].strip())
                kruskal_results[current_layer][model_name] = statistic
    
    return kruskal_results, model_names

def plot_kruskal_wallis_results(kruskal_results, model_names, output_path='kruskal_wallis_plot.png'):
    """
    Plot the Kruskal-Wallis H-statistic for each layer across models and save the plot.
    """
    num_layers = len(kruskal_results)
    layers = np.arange(num_layers)

    plt.figure(figsize=(12, 8))
    
    for model_name in model_names:
        statistics = [kruskal_results[layer][model_name] for layer in layers]
        plt.plot(layers, statistics, label=model_name, marker='o')
    
    plt.title('Kruskal-Wallis H-Statistic for Each Layer Across Models', fontsize=20)
    plt.xlabel('Layer Number', fontsize=18)
    plt.ylabel('Kruskal-Wallis H-Statistic', fontsize=18)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.yscale('log')  # Use log scale for better visualization
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def compute_model_wise_mean_and_std(kruskal_results, model_names):
    """
    Compute Mean and standard deviation of H-Statistics across layers for each model.
    """
    statistics = np.array([[kruskal_results[layer][model] for layer in range(len(kruskal_results))] for model in model_names])

    mean = np.mean(statistics, axis=1)
    std = np.std(statistics, axis=1)
    
    return mean, std

# Read Kruskal-Wallis results from the saved text file
kruskal_results, model_names = read_kruskal_wallis_results('features/ks_model_comparison_results.txt')

# Generate and save the plot
plot_kruskal_wallis_results(kruskal_results, model_names, output_path='features/kruskal_wallis_results_plotA.png')

# Compute Mean and STD
mean_results, std_results = compute_model_wise_mean_and_std(kruskal_results, model_names)

# Print and save results
with open('features/kruskal_wallis_mean_std_resultsA.txt', 'w') as file:
    for i, model_name in enumerate(model_names):
        result = f"{model_name} Mean: {mean_results[i]:.4f}, STD: {std_results[i]:.4f}\n"
        print(result)
        file.write(result)
        


import os
import numpy as np


def read_kruskal_wallis_results(filename):
    kruskal_results = {}
    model_names = []
    
    with open(filename, 'r') as file:
        current_layer = None
        for line in file:
            if line.startswith("Layer"):
                current_layer = int(line.split()[1])
                kruskal_results[current_layer] = {}
            elif "vs full_fine_tune:" in line:
                model_name = line.split("vs")[0].strip()
                if model_name not in model_names:
                    model_names.append(model_name)
            elif "Statistic:" in line:
                statistic = float(line.split(":")[1].strip())
                kruskal_results[current_layer][model_name] = {'statistic': statistic}
            elif "p-value:" in line:
                p_value = float(line.split(":")[1].strip())
                kruskal_results[current_layer][model_name]['p_value'] = p_value
    
    return kruskal_results, model_names

def plot_kruskal_wallis_results(kruskal_results, model_names, output_folder):
    num_layers = len(kruskal_results)
    layers = np.arange(num_layers)

    # Plot H-statistics
    plt.figure(figsize=(12, 6))
    for model_name in model_names:
        statistics = [kruskal_results[layer][model_name]['statistic'] for layer in layers]
        plt.plot(layers, statistics, marker='o', label=model_name)
    
    plt.title('Kruskal-Wallis H-Statistic for Each Layer Across Models', fontsize=16)
    plt.xlabel('Layer Number', fontsize=14)
    plt.ylabel('H-Statistic', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.yscale('log')
    plt.savefig(os.path.join(output_folder, 'kruskal_wallis_h_statistics.png'), dpi=300)
    plt.close()

    # Plot p-values
    plt.figure(figsize=(12, 6))
    for model_name in model_names:
        p_values = [kruskal_results[layer][model_name]['p_value'] for layer in layers]
        plt.plot(layers, p_values, marker='o', label=model_name)
    
    plt.title('Kruskal-Wallis p-values for Each Layer Across Models', fontsize=16)
    plt.xlabel('Layer Number', fontsize=14)
    plt.ylabel('p-value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.yscale('log')
    plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
    plt.savefig(os.path.join(output_folder, 'kruskal_wallis_p_values.png'), dpi=300)
    plt.close()

def compute_mad_std(kruskal_results, model_names):
    statistics = {model: [] for model in model_names}
    for layer in kruskal_results:
        for model in model_names:
            statistics[model].append(kruskal_results[layer][model]['statistic'])
    
    mad_std = {}
    for model in model_names:
        values = np.array(statistics[model])
        mad = np.mean(np.abs(values - 0))  # MAD from zero
        std = np.sqrt(np.mean(values**2))  # STD from zero (root mean square)
        mad_std[model] = {'MAD': mad, 'STD': std}
    
    return mad_std

# Main execution
input_file = 'features/ks_model_comparison_results.txt'
output_folder = 'features'

# Read the results
kruskal_results, model_names = read_kruskal_wallis_results(input_file)

# Generate plots
plot_kruskal_wallis_results(kruskal_results, model_names, output_folder)

# Compute MAD and STD
mad_std_results = compute_mad_std(kruskal_results, model_names)

# Save MAD and STD results
with open(os.path.join(output_folder, 'kruskal_wallis_mad_std.txt'), 'w') as f:
    for model, values in mad_std_results.items():
        f.write(f"{model} MAD: {values['MAD']:.4f}, STD: {values['STD']:.4f}\n")






import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


def create_heatmap(kruskal_results, model_names, output_folder):
    # Prepare data for heatmap
    data = []
    for layer in range(len(kruskal_results)):
        row = [kruskal_results[layer][model]['statistic'] for model in model_names]
        data.append(row)
    
    df = pd.DataFrame(data, columns=model_names)
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df, annot=True, fmt='.2f', cmap='YlOrRd', 
                linewidths=0.5, cbar_kws={'label': 'H-statistic'})
    
    plt.title('Kruskal-Wallis H-statistic Heatmap\nComparing Model Variants Across Layers', fontsize=16)
    plt.xlabel('Models', fontsize=14)
    plt.ylabel('Layers', fontsize=14)
    plt.tight_layout()
    
    # Save the heatmap
    plt.savefig(os.path.join(output_folder, 'kruskal_wallis_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(df.describe())

    # Identify layers with highest and lowest H-statistics
    max_value = df.max().max()
    min_value = df.min().min()
    max_layer, max_model = np.unravel_index(df.values.argmax(), df.shape)
    min_layer, min_model = np.unravel_index(df.values.argmin(), df.shape)

    print(f"\nHighest H-statistic: Layer {max_layer}, Model: {df.columns[max_model]}, Value: {max_value:.2f}")
    print(f"Lowest H-statistic: Layer {min_layer}, Model: {df.columns[min_model]}, Value: {min_value:.2f}")


# Create heatmap
create_heatmap(kruskal_results, model_names, output_folder)











