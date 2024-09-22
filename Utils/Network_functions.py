
## Python standard libraries
from __future__ import print_function
from __future__ import division


from Utils.Feature_Extraction_Layer import Feature_Extraction_Layer

from src.models import ASTModel

def initialize_model(model_name, num_classes, numBins, sample_rate=16000,
                     t_mode='full_fine_tune', histogram=True, h_shared=True, a_shared=True,
                     parallel=True, use_pretrained=True,
                     input_feature='STFT', RGB=True,
                     window_length=1024, hop_length=128, number_mels=64,
                     adapter_location='ffn', adapter_mode='parallel', 
                     histogram_location='ffn', histogram_mode='parallel'):
    
    if model_name == "AST":
        RGB = False    

    # Initialize feature layer
    feature_layer = Feature_Extraction_Layer(input_feature=input_feature, sample_rate=sample_rate,
                                             window_length=window_length, hop_length=hop_length, number_mels=number_mels, RGB=RGB)
    
    ft_dims = feature_layer.output_dims

    model_ft = None
    inpf = ft_dims[1]
    inpt = ft_dims[2]
    print(f'feature shape f by t: {inpf} by {inpt}\n')
    Use_A = (t_mode == 'adapters')

    h_mode = histogram
    if h_mode:
        Use_H = True
        h_operation = 'add'
    else:
        Use_H = False
        h_operation = None
        histogram_mode = None
        histogram_location = None
        
    a_shared = a_shared
    h_shared = h_shared
    NumBins = numBins
    
    
    model_ft = ASTModel(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt, 
            imagenet_pretrain=use_pretrained, audioset_pretrain=use_pretrained, adapter_shared=a_shared, hist_shared=h_shared,
            use_adapters=Use_A, adapter_mode=adapter_mode, adapter_location=adapter_location,NumBins=NumBins,
            use_histogram=Use_H, histogram_mode=histogram_mode, histogram_operation=h_operation, histogram_location=histogram_location)

    print("\nSettings:")
    print(f"Use adapters: {Use_A}")
    print(f"Adapter mode: {adapter_mode}")
    print(f"Adapter location: {adapter_location}")
    print(f"Adapter shared: {a_shared}")
    print(f"Use histogram: {Use_H}")
    print(f"Histogram mode: {histogram_mode}")
    print(f"Histogram location: {histogram_location}")
    print(f"Histogram shared: {h_shared}\n")
        
    if t_mode == 'full_fine_tune':
        for param in model_ft.parameters():
            param.requires_grad = True
    elif t_mode == 'linear_probing':
        for param in model_ft.parameters():
            param.requires_grad = False
        for param in model_ft.mlp_head.parameters():
            param.requires_grad = True
    elif t_mode == 'adapters':
        for param in model_ft.parameters():
            param.requires_grad = False

        if adapter_location in ['mhsa', 'mhsa_ffn', 'mhsa_out', 'all']:
            for param in model_ft.adapters_mhsa.parameters():
                param.requires_grad = True
        if adapter_location in ['ffn', 'mhsa_ffn', 'ffn_out', 'all']:
            for param in model_ft.adapters_ffn.parameters():
                param.requires_grad = True
        if adapter_location in ['out', 'mhsa_out', 'ffn_out', 'all']:
            for param in model_ft.adapters_out.parameters():
                param.requires_grad = True

        for param in model_ft.mlp_head.parameters():
            param.requires_grad = True

    if Use_H:
        if histogram_location in ['mhsa', 'mhsa_ffn', 'mhsa_out', 'all']:
            for param in model_ft.histogram_layers_mhsa.parameters():
                param.requires_grad = True
        if histogram_location in ['ffn', 'mhsa_ffn', 'ffn_out', 'all']:
            for param in model_ft.histogram_layers_ffn.parameters():
                param.requires_grad = True
        if histogram_location in ['out', 'mhsa_out', 'ffn_out', 'all']:
            for param in model_ft.histogram_layers_out.parameters():
                param.requires_grad = True    
    

    return model_ft, feature_layer


