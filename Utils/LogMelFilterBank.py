import torch.nn as nn
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

class MelSpectrogramExtractor(nn.Module): 
    def __init__(self, sample_rate=16000, n_fft=512, win_length=512, hop_length=160, n_mels=64, fmin=50, fmax=8000):
        super(MelSpectrogramExtractor, self).__init__()
        
        # Settings for Spectrogram
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        
        self.spectrogram_extractor = Spectrogram(n_fft=win_length, hop_length=hop_length, 
                                                  win_length=win_length, window=window, center=center, 
                                                  pad_mode=pad_mode, 
                                                  freeze_parameters=True)

        ref = 1.0
        amin = 1e-10
        top_db = None
        
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=win_length, 
            n_mels=n_mels, fmin=fmin, fmax=fmax, ref=ref, amin=amin, top_db=top_db, freeze_parameters=True)
        
        t_n = n_mels

        #scale_factor = sample_rate / 16000.0
        #time_drop_width = int(t_n * scale_factor * 0.125)
        
        # Spec augmenter
        #self.spec_augmenter = SpecAugmentation(time_drop_width=time_drop_width, time_stripes_num=2, 
        #    freq_drop_width=16, freq_stripes_num=2)

        self.bn0 = nn.BatchNorm2d(t_n)

    def forward(self, waveform):

        spectrogram = self.spectrogram_extractor(waveform)
        log_mel_spectrogram = self.logmel_extractor(spectrogram)

        log_mel_spectrogram = log_mel_spectrogram.transpose(1, 3)
        log_mel_spectrogram = self.bn0(log_mel_spectrogram)
        log_mel_spectrogram = log_mel_spectrogram.transpose(1, 3)
        
        # if self.training:
        #     log_mel_spectrogram = self.spec_augmenter(log_mel_spectrogram)
            
        log_mel_spectrogram = log_mel_spectrogram.squeeze(1).transpose(1, 2)

        return log_mel_spectrogram

