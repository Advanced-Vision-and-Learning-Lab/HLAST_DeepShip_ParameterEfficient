import torch.nn as nn
from nnAudio import features
import torch
from .LogMelFilterBank import MelSpectrogramExtractor  
import pdb

class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, sample_rate=16000, window_length=250, 
                 hop_length=64, RGB=False, spec_norm=False):
        super(Feature_Extraction_Layer, self).__init__()
        
        self.sample_rate = sample_rate   
        self.sample_frequency = sample_rate 
        
        self.num_channels = 1
        
        self.spec_norm = spec_norm
        self.input_feature = input_feature
        
        # Initialize logmelfbank
        n_fft = 1024
        win_length = 1024
        hop_length = 1000 
        n_mels = 48
        fmin = 1
        fmax = 8000
        
        # Initialize logmelfbank
        # n_fft = 4096
        # win_length = 4096
        # hop_length = 512 
        # n_mels = 128
        # fmin = 1
        # fmax = 4000
        
        self.LogMelFBank = MelSpectrogramExtractor(
            sample_rate=sample_rate, 
            n_fft=n_fft,
            win_length=win_length, 
            hop_length=hop_length, 
            n_mels=n_mels,
            fmin=fmin, 
            fmax=fmax
        )

        self.features = {'LogMelFBank': self.LogMelFBank}
        
        
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
            
        
    def forward(self, x):
       
        #Extract audio feature
        x = self.features[self.input_feature](x)

        x = x.unsqueeze(1)


        return x

        
