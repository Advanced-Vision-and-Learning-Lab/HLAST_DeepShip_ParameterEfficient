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
        self.dataset_mean = 48.26377
        self.dataset_std = 11.56119
        self.input_feature = input_feature
        
        # Initialize logmelfbank
        # Scale factors based on the ratio of the new sample rate to the original sample rate (16000)
        scale_factor = sample_rate / 16000.0
        
        n_fft = 1024
        win_length = 1024
        hop_length = 1000 
        n_mels = 48
        fmin = 10
        fmax = 8000

        # fmin = max(1e-2, int(50 * scale_factor))
        # fmax = min(int(8000 * scale_factor), sample_rate // 2)
        
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
        #pdb.set_trace()
        
        # Normalize the feature
        #if self.spec_norm:
        x = (x - self.dataset_mean) / (self.dataset_std) 
        x *= 0.5  


        x = x.unsqueeze(1)
        # Repeat channel dimension if needed (e.g., for CNNs)
        #x = x.repeat(1, self.num_channels, 1, 1)  


        return x

        
