import torch.nn as nn
from nnAudio import features
import torch
from .LogMelFilterBank import MelSpectrogramExtractor  
import pdb

class Feature_Extraction_Layer(nn.Module):
    def __init__(self, input_feature, sample_rate=16000, window_length=250, 
                 hop_length=64, RGB=False, spec_norm=False, frame_shift=10.0):
        super(Feature_Extraction_Layer, self).__init__()
        
        self.sample_rate = sample_rate   
        self.sample_frequency = sample_rate 
        self.frame_shift = frame_shift  
        
        self.spec_norm = spec_norm
        self.dataset_mean = 47.69
        self.dataset_std = 10.76
        
        
        window_length = window_length
        hop_length = hop_length
        #Convert window and hop length to ms
        window_length /= 1000
        hop_length /= 1000
        
        
        if RGB:
            num_channels = 3
            MFCC_padding = nn.ZeroPad2d((3,2,16,16))
        else:
            num_channels = 1
            MFCC_padding = nn.ZeroPad2d((1,4,0,0))
        
        self.num_channels = num_channels
        self.input_feature = input_feature

        #Return Mel Spectrogram that is 48 x 48
        self.Mel_Spectrogram = nn.Sequential(features.mel.MelSpectrogram(sample_rate,n_mels=40,win_length=int(window_length*sample_rate),
                                            hop_length=int(hop_length*sample_rate),
                                            n_fft=int(window_length*sample_rate), verbose=False), nn.ZeroPad2d((1,0,8,0)))
                                            

    
    
        #Return MFCC that is 16 x 48 (TDNN models) or 48 x 48 (CNNs)
        self.MFCC = nn.Sequential(features.mel.MFCC(sr=sample_rate, n_mfcc=16, 
                                        n_fft=int(window_length*sample_rate), 
                                                win_length=int(window_length*sample_rate), 
                                                hop_length=int(hop_length*sample_rate),
                                                n_mels=48, center=False, verbose=False), MFCC_padding)

        #Return STFT that is 48 x 48
        self.STFT = nn.Sequential(features.STFT(sr=sample_rate,n_fft=int(window_length*sample_rate), 
                                        hop_length=int(hop_length*sample_rate),
                                        win_length=int(window_length*sample_rate), trainable=False,
                                        output_format='Magnitude',
                                        freq_bins=48,verbose=False), nn.ZeroPad2d((1,0,0,0)))
    
        
        #Return GFCC that is 64 x 48
        self.GFCC = nn.Sequential(features.Gammatonegram(sr=sample_rate,
                                                hop_length=int(hop_length*sample_rate),
                                                n_fft=int(window_length*sample_rate),
                                                verbose=False,n_bins=64), nn.ZeroPad2d((1,0,0,0)))
        

        #Return CQT that is 64 x 48
        self.CQT = nn.Sequential(features.CQT(sr=sample_rate, n_bins=64, 
                                        hop_length=int(hop_length*sample_rate),
                                        verbose=False), nn.ZeroPad2d((1,0,0,0)))
        
        #Return VQT that is 64 x 48
        self.VQT = nn.Sequential(features.VQT(sr=sample_rate,hop_length=int(hop_length*sample_rate),
                                        n_bins=64,earlydownsample=False,verbose=False), nn.ZeroPad2d((1,0,0,0)))
        

        
        # Initialize logmelfbank
        # Scale factors based on the ratio of the new sample rate to the original sample rate (16000)
        scale_factor = sample_rate / 16000.0
        scale_factor = 1
        # Calculate the new parameters
        # n_fft = int(512 * scale_factor)
        # win_length = int(512 * scale_factor)
        # hop_length = int(160 * scale_factor)
        # n_mels = 64
        
        
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

        self.features = {'Mel_Spxectrogram': self.Mel_Spectrogram, 
                         'MFCC': self.MFCC, 'STFT': self.STFT,
                         'GFCC': self.GFCC, 'CQT': self.CQT, 'VQT': self.VQT,
                         'LogMelFBank': self.LogMelFBank}
        
        
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


        #x = x.unsqueeze(1)
        #Repeat channel dimension if needed (CNNs)
        #x = x.repeat(1, self.num_channels,1,1)

        return x

        
