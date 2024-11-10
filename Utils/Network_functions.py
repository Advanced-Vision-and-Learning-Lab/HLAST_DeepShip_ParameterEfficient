
## Python standard libraries
from __future__ import print_function
from __future__ import division
import pdb
from Utils.Feature_Extraction_Layer import Feature_Extraction_Layer

from src.models.ast_base import ASTBase
from src.models.ast_linear_probe import ASTLinearProbe
from src.models.ast_adapter import ASTAdapter
from src.models.ast_histogram import ASTHistogram

def initialize_model(model_name, num_classes, numBins, RR, sample_rate=16000,segment_length=5,
                     t_mode='full_fine_tune', h_shared=True, a_shared=True,
                     parallel=True, input_feature='STFT', RGB=True,
                     window_length=512, hop_length=256, number_mels=64,
                     adapter_location='ffn', adapter_mode='parallel', 
                     histogram_location='ffn', histogram_mode='parallel'):
    
    if model_name == "AST":
        RGB = False    

    # Initialize feature layer
    feature_layer = Feature_Extraction_Layer(input_feature=input_feature, sample_rate=sample_rate,segment_length=segment_length,
                                             window_length=window_length, hop_length=hop_length, number_mels=number_mels, RGB=RGB)
    
    ft_dims = feature_layer.output_dims
    inpf, inpt = ft_dims[1], ft_dims[2]
    print(f'feature shape f by t: {inpf} by {inpt}')

    # Initialize the appropriate model
    if t_mode == 'full_fine_tune':
        model_ft = ASTBase(num_labels=num_classes, max_length=inpt, num_mel_bins=inpf)
    elif t_mode == 'linear_probing':
        model_ft = ASTLinearProbe(num_labels=num_classes, max_length=inpt, num_mel_bins=inpf)
    elif t_mode == 'adapters':
        model_ft = ASTAdapter(num_labels=num_classes, max_length=inpt, num_mel_bins=inpf,
                              adapter_size=RR, adapter_location=adapter_location,
                              adapter_shared=a_shared, adapter_mode=adapter_mode)
    elif t_mode == 'histogram':
        model_ft = ASTHistogram(num_labels=num_classes, max_length=inpt, num_mel_bins=inpf,
                                num_bins=numBins, histogram_location=histogram_location,
                                hist_shared=h_shared, histogram_mode=histogram_mode)
    else:
        raise ValueError(f"Unknown training mode: {t_mode}")

    print("\nSettings:")
    print(f"Training mode: {t_mode}")
    print(f"Use adapters: {t_mode == 'adapters'}")
    print(f"Adapter mode: {adapter_mode}")
    print(f"Adapter location: {adapter_location}")
    print(f"Adapter shared: {a_shared}")
    print(f"Histogram mode: {histogram_mode}")
    print(f"Histogram location: {histogram_location}")
    print(f"Histogram shared: {h_shared}\n")

    # Set requires_grad for parameters
    if t_mode == 'full_fine_tune':
        for param in model_ft.parameters():
            param.requires_grad = True

    return model_ft, feature_layer
