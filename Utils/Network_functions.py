from Utils.Feature_Extraction_Layer import Feature_Extraction_Layer
from src.models.ast_base import ASTBase
from src.models.ast_linear_probe import ASTLinearProbe
from src.models.ast_adapter import ASTAdapter
from src.models.ast_histogram import ASTHistogram
from src.models.ast_bias import ASTBias

from src.models.ast_lora import ASTLoRA

def initialize_model(model_name, num_classes, numBins, RR, sample_rate=16000,segment_length=5,
                     t_mode='full_fine_tune', h_shared=True, a_shared=True,
                     parallel=True, input_feature='STFT',
                     window_length=512, hop_length=256, number_mels=64,
                     adapter_location='ffn', adapter_mode='parallel', 
                     histogram_location='ffn', histogram_mode='parallel',
                     lora_target='q', lora_rank=4, r_shared=False, lora_alpha=1.0):

    # Initialize feature layer
    feature_layer = Feature_Extraction_Layer(input_feature=input_feature, sample_rate=sample_rate,segment_length=segment_length,
                                             window_length=window_length, hop_length=hop_length, number_mels=number_mels)
    
    ft_dims = feature_layer.output_dims
    inpf, inpt = ft_dims[1], ft_dims[2]
    print(f'feature shape f by t: {inpf} by {inpt}')

    # Initialize the appropriate model
    if t_mode == 'full_fine_tune':
        model_ft = ASTBase(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt)
    elif t_mode == 'linear_probing':
        model_ft = ASTLinearProbe(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt)
    elif t_mode == 'adapters':
        model_ft = ASTAdapter(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt,
                              RR=RR, adapter_location=adapter_location,
                              adapter_shared=a_shared, adapter_mode=adapter_mode)
    elif t_mode == 'histogram':
        model_ft = ASTHistogram(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt,
                                NumBins=numBins, histogram_location=histogram_location,
                                hist_shared=h_shared, histogram_mode=histogram_mode)
    elif t_mode == 'bias':                            
        	model_ft = ASTBias(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt)
        
    # 'q' or 'qv'
    elif t_mode == 'lora':    
        model_ft = ASTLoRA(label_dim=num_classes, input_fdim=inpf, input_tdim=inpt,
                lora_target=lora_target, lora_rank=lora_rank, lora_alpha=lora_alpha, lora_shared=r_shared)
        
    else:
        raise ValueError(f"Unknown training mode: {t_mode}")

    print("\n\nSettings:")
    print(f"Training mode: {t_mode}")
    print(f"Adapter mode: {adapter_mode}")
    print(f"Adapter location: {adapter_location}")
    print(f"Adapter shared: {a_shared}")
    print(f"Histogram mode: {histogram_mode}")
    print(f"Histogram location: {histogram_location}")
    print(f"Histogram shared: {h_shared}")
    print(f"lora: {lora_target}, rank: {lora_rank}, shared: {r_shared}\n\n")
    
    # Set requires_grad for parameters
    if t_mode == 'full_fine_tune':
        for param in model_ft.parameters():
            param.requires_grad = True
            

    return model_ft, feature_layer
