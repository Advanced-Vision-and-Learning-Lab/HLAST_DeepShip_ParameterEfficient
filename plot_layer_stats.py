import matplotlib.pyplot as plt
import numpy as np

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
cosine_similarities, model_names = read_cosine_similarity('features/cosine_similarity_results_2.txt')

# Generate and save the plot
plot_cosine_similarity(cosine_similarities, model_names, output_path='features/cosine_similarity_plot_2.png')


flattened_values = [value for sublist in cosine_similarities.values() for value in sublist[1:]]
cosine_similarities_reshaped = np.array(flattened_values).reshape(12, 3)


def compute_layer_wise_mad(z_scores):
    """ 
    Compute Mean Absolute Deviation (MAD) of Z-Scores from 1 for each layer compared to the full fine-tune model. 
    The MAD is calculated layer-wise and then averaged across all layers for each model.
    """
    # Subtract 1 from each z-score
    deviation_from_1 = np.abs(z_scores - 1)
    
    # Compute the mean absolute deviation across layers for each model
    mad = np.mean(deviation_from_1, axis=0)
    
    return mad

mad_results = compute_layer_wise_mad(cosine_similarities_reshaped)

# Print results
model_names = ['Linear Probing', 'Histogram', 'Adapters']
for i, model_name in enumerate(model_names):
    print(f"{model_name} MAD compared to Full Fine-Tune: {mad_results[i]:.4f}\n")

# Save the results to a file
with open('features/mad_layer_comparison_cosine.txt', 'w') as file:
    for i, model_name in enumerate(model_names):
        file.write(f"{model_name} MAD compared to Full Fine-Tune: {mad_results[i]:.4f}\n")

