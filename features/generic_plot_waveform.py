''' Simple script to plot waveforms '''
import numpy as np
from torchvision import transforms
from nnAudio import features
import torch 
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import argparse


class SplitWin_Normalize(object):
    def __call__(self, signal):
        ''' Multi-pass split-window normalizer '''
        # data comes in as (channel, frequency, time)
        data_in=np.array(signal[0])
        data_in = data_in.transpose(1,0)
        win_sizes=[41, 41, 25]
        gaps=[20, 20, 0]
        axis=-1
        clips=[3, 2, 1]
        replaces=[1, 1, 1]
        num_lines, num_bins = data_in.shape

        assert(
            len(win_sizes) == len(gaps) == len(clips) == len(replaces)
        ), "normalizer parameter lengths must be equal"

        smoothed = np.zeros((num_lines, num_bins))
        clipped = data_in.copy()

        for winsz, gap, clip, replace in zip(win_sizes, gaps, clips, replaces):

            if gap == 0:
                # smoothing filter
                clipped = ndimage.filters.uniform_filter1d(
                    clipped, winsz, axis=axis, mode="reflect"
                )
            # split win + clip/replace, filter defaults to last axis
            else:
                filtered = ndimage.filters.uniform_filter1d(
                    clipped, winsz, axis=axis, mode="reflect"
                )

                if axis == 0:
                    fwd_idx = (np.arange(0, num_lines) + gap + winsz / 2 + 1).astype(int)
                    fwd_idx = np.where(
                        fwd_idx < num_lines, fwd_idx, 2 * num_lines - 2 - fwd_idx
                    )
                    aft_idx = (np.abs(np.arange(0, num_lines) - gap - winsz / 2 - 1)).astype(int)

                    smoothed = (filtered[fwd_idx, :] + filtered[aft_idx,:]) / 2.0
                
                # else axis == 1 or axis == -1
                else:
                    fwd_idx = (np.arange(0, num_bins) + gap + winsz / 2 + 1).astype(int)
                    fwd_idx = np.where(
                        fwd_idx < num_bins, fwd_idx, 2 * num_bins - 2 - fwd_idx
                    ).astype(int)
                    aft_idx = (np.abs(np.arange(0, num_bins) - gap - winsz / 2 - 1)).astype(int)
                    smoothed = (filtered[:, fwd_idx] + filtered[:, aft_idx]) / 2.0
                
                clipped = np.where(clipped > (clip * smoothed), replace * smoothed, clipped)
        
        data_out = data_in / clipped
        data_out = data_out.transpose(1,0)
        return torch.Tensor(data_out).unsqueeze(0)

class Normalize(object):
    '''custom normalize from -1 to 1'''
    def __call__(self, signal):
        signal = torch.Tensor(signal)
        res = (2*(signal - torch.min(signal)) / (torch.max(signal) - torch.min(signal))) - 1
        return res 
    
class TruncateFrequencies(object):
    '''Truncate Frequencies'''

    def __init__(self, n_mels, sample_rate, mel, upper_limit, lower_limit):
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.mel = mel
        self.upper_limit = upper_limit
        self.lower_limit = lower_limit

    def __call__(self, signal):
        f = np.linspace(0, self.sample_rate / 2, self.n_mels)
        if self.mel:
            signal = signal.unsqueeze(0)
            f = 2595.0 * np.log10(1.0 + f / 700.0)
        # limit to frequencies
        f1 = np.where(f>self.lower_limit)[0][0]
        f2 = np.where(f<self.upper_limit)[0][-1]

        res = signal[:,f1:f2,:]
        return res

class Log10(object):
    def __call__(self, signal):
        return 10*np.log10(signal)
    
class STD(object):
    def __call__(self, signal):
        std_sig = signal / signal.std()

        vmin = 0
        vmax = 3
        res = np.clip(std_sig, vmin, vmax)

        return torch.Tensor(res)
    
def get_audio(sample_rate, win_len, hop_len, n_bins, upper_limit, lower_limit): 
    num_channels = 1
    
    signal_transform = features.STFT(sr=sample_rate,n_fft=win_len, 
                                        hop_length=hop_len,
                                        win_length=win_len, 
                                        output_format='Magnitude',
                                        freq_bins=n_bins,verbose=False)
    
    transform_stft_log = transforms.Compose([
            Normalize(),
            signal_transform,
            Log10(),
            TruncateFrequencies(n_mels=n_bins, sample_rate=sample_rate, mel=False, upper_limit=upper_limit, lower_limit=lower_limit),
            transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
        ])

    transform_stft_sw_log = transforms.Compose([
            Normalize(),
            signal_transform,
            SplitWin_Normalize(),
            Log10(),
            TruncateFrequencies(n_mels=n_bins, sample_rate=sample_rate, mel=False, upper_limit=upper_limit, lower_limit=lower_limit),
            transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
        ])
    
    transform_stft_sw_log_std = transforms.Compose([
            Normalize(),
            signal_transform,
            SplitWin_Normalize(),
            Log10(),
            STD(),
            transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
        ])

    transform_stft_sw_log_std_trunc = transforms.Compose([
            Normalize(),
            signal_transform,
            SplitWin_Normalize(),
            Log10(),
            STD(),    
            TruncateFrequencies(n_mels=n_bins, sample_rate=sample_rate, mel=False, upper_limit=upper_limit, lower_limit=lower_limit),
            transforms.Lambda(lambda x: x.repeat(num_channels,1,1)),
        ])
    
    return transform_stft_log, transform_stft_sw_log, transform_stft_sw_log_std, transform_stft_sw_log_std_trunc


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-w', '--wavfile', 
                        default='wavfile.wav',
                        help="Enter path to wavfile")

    args = parser.parse_args()

    wavfile_path = args.wavfile

    sr, wav = wavfile.read(wavfile_path)

    ## parameters

    # frequency truncation upper/lower limits in Hz
    upper_lim = 1000
    lower_lim = 10

    window_length = int(sr/2)
    hop_length = int(window_length/2)
    n_bins = int(window_length/2) + 1

    print(f"window_length: {window_length}")
    print(f"hop_length: {hop_length}")
    print(f"n_bins: {n_bins}")
    print(f"sample rate: {sr}")
    print(f"upper_limit: {upper_lim}")
    print(f"lower_limit: {lower_lim}")

    # get transforms
    transform_stft, transform_stft_sw_log, transform_stft_sw_log_std, transform_stft_sw_log_std_trunc = get_audio(sr, window_length, hop_length, n_bins, upper_lim, lower_lim)

    # Add all transforms (and titles/None) to be plotted
    transforms_to_plot = [
        ("STFT, LOG10, Truncate", transform_stft(wav)), 
        ("STFT, NSE, LOG10, Truncate", transform_stft_sw_log(wav)), 
        ("STFT, NSE, LOG10, STD", transform_stft_sw_log_std(wav)), 
        ("STFT, NSE, LOG10, STD, Truncate", transform_stft_sw_log_std_trunc(wav)),
    ]

    # create figure with x number of subplots based off transforms_to_plot
    fig, axes = plt.subplots(len(transforms_to_plot), figsize=(20, 20))
    fig.subplots_adjust(wspace=.4, hspace=.6)

    fig.suptitle(f"SR={sr}, WIN_LEN={window_length}, HOP_LEN={hop_length}, NFFT={window_length}, N_BINS={n_bins}, LOWER_CLIP_LIM={lower_lim}, UPPER_CLIP_LIM={upper_lim}")

    for (title, transform), axis in zip(transforms_to_plot, axes):
        im = axis.imshow(transform.cpu().squeeze(0).transpose(1,0), aspect='auto', interpolation='none', origin='upper', cmap='gray_r')
        fig.colorbar(im, ax=axis, format="%+2.0f dB")
        if title is not None:
            axis.set_title(title)
    
    print(wavfile_path)
    plt.show()

    
