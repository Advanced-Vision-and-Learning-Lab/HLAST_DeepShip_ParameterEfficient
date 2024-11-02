import torch.nn as nn
import torch
from .LogMelFilterBank import MelSpectrogramExtractor  
import pdb
from nnAudio import features
#from .progressive_tokenization import ProgressiveTokenizationModule

class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, sample_rate=16000, window_length=4096, 
                 hop_length=512, number_mels=64, RGB=False):
        super(Feature_Extraction_Layer, self).__init__()
        
        self.sample_rate = sample_rate   
        self.sample_frequency = sample_rate 
        self.num_channels = 1
        self.input_feature = input_feature
        
    
        # Initialize logmelfbank
        win_length = window_length
        n_fft = window_length
        hop_length = hop_length 
        n_mels = number_mels
        fmin = 1
        fmax = 8000
        
        self.LogMelFBank = MelSpectrogramExtractor(
            sample_rate=sample_rate, 
            n_fft=n_fft,
            win_length=win_length, 
            hop_length=hop_length, 
            n_mels=n_mels,
            fmin=fmin, 
            fmax=fmax
        )

        # Initialize nnAudio MelSpectrogram
        self.Mel_Spectrogram = features.mel.MelSpectrogram(
            sr=sample_rate,
            n_mels=n_mels,
            win_length=win_length,
            hop_length=hop_length,
            n_fft=n_fft,
            verbose=False
        )

        self.features = {'LogMelFBank': self.LogMelFBank, 'MelSpec': self.Mel_Spectrogram}
                

        #self.progressive_tokenization = ProgressiveTokenizationModule(input_channels=1)  


        self.output_dims = None
        self.calculate_output_dims()

    
    def calculate_output_dims(self):
        try:
            length_in_seconds = 5  
            samples = int(self.sample_rate * length_in_seconds)
            dummy_input = torch.randn(1, samples)  
            with torch.no_grad():
                output = self.features[self.input_feature](dummy_input)
                self.output_dims = output.shape
        except Exception as e:
            print(f"Failed to calculate output dimensions: {e}\n")
            self.output_dims = None
            
            
    # def calculate_output_dims(self):
    #     try:
    #         length_in_seconds = 5  
    #         samples = int(self.sample_rate * length_in_seconds)
    #         dummy_input = torch.randn(1, samples)  
    #         with torch.no_grad():
    #             # Get the output from the feature extraction
    #             feature_output = self.features[self.input_feature](dummy_input)
    #             feature_output = feature_output.unsqueeze(1)  # Add channel dimension
    
    #             # Pass through the progressive tokenization module
    #             tokenized_output = self.progressive_tokenization(feature_output)
    
    #             # Reshape the output to combine batch and channel dimensions
    #             reshaped_output = tokenized_output.view(-1, tokenized_output.shape[2], tokenized_output.shape[3])
    
    #             # Update the output dimensions
    #             self.output_dims = reshaped_output.shape
    
    #             print(f"Calculated output dimensions: {self.output_dims}")
    #             print(f"feature shape f by t: {self.output_dims[1]} by {self.output_dims[2]}")
    #     except Exception as e:
    #         print(f"Failed to calculate output dimensions: {e}")
    #         self.output_dims = None

    def forward(self, x):
        
        #Extract audio feature
        x = self.features[self.input_feature](x)

        x = x.unsqueeze(1)

        # Apply progressive tokenization
        #x = self.progressive_tokenization(x)


        return x

