import torch
import torch.nn as nn
from transformers import ASTModel
from .RBFHistogramPooling import HistogramLayer

class ASTHistogram(nn.Module):
    def __init__(self, num_labels, max_length, num_mel_bins, num_bins,
                 histogram_location='mhsa_ffn', hist_shared=True, histogram_mode='parallel', 
                 model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        self.model = ASTModel.from_pretrained(model_ckpt, max_length=max_length, num_mel_bins=num_mel_bins, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.original_embedding_dim = self.model_config.hidden_size
        self.histogram_location = histogram_location
        self.histogram_mode = histogram_mode
        self.hist_shared = hist_shared
        self.num_bins = num_bins
        
        self.histogram_layers = nn.ModuleDict()
        num_layers = len(self.model.encoder.layer)
        
        print(f"Initializing ASTHistogram with the following configuration:")
        print(f"Histogram location: {histogram_location}")
        print(f"Histogram mode: {histogram_mode}")
        print(f"Histogram shared: {hist_shared}")
        print(f"Number of bins: {self.num_bins}")
        print(f"Number of transformer layers: {num_layers}")
        
        if hist_shared:
            if 'mhsa' in histogram_location:
                self.histogram_layers['mhsa'] = HistogramLayer(in_channels=self.original_embedding_dim,
                                                               kernel_size=1, dim=1,
                                                               num_bins=num_bins, stride=1,
                                                               normalize_count=True, normalize_bins=True)
                output_size = int(self.original_embedding_dim / self.histogram_layers['mhsa'].bin_widths_conv.out_channels)
                self.histogram_layers['mhsa'].hist_pool = nn.AdaptiveAvgPool1d(output_size)
                print(f"Added shared MHSA histogram layer with {num_bins} bins and output size {output_size}")
            if 'ffn' in histogram_location:
                self.histogram_layers['ffn'] = HistogramLayer(in_channels=self.original_embedding_dim,
                                                              kernel_size=1, dim=1,
                                                              num_bins=num_bins, stride=1,
                                                              normalize_count=True, normalize_bins=True)
                output_size = int(self.original_embedding_dim / self.histogram_layers['ffn'].bin_widths_conv.out_channels)
                self.histogram_layers['ffn'].hist_pool = nn.AdaptiveAvgPool1d(output_size)
                print(f"Added shared FFN histogram layer with {num_bins} bins and output size {output_size}")
        else:
            if 'mhsa' in histogram_location:
                self.histogram_layers['mhsa'] = nn.ModuleList([HistogramLayer(in_channels=self.original_embedding_dim,
                                                                              kernel_size=1, dim=1,
                                                                              num_bins=num_bins, stride=1,
                                                                              normalize_count=True, normalize_bins=True) 
                                                               for _ in range(num_layers)])
                for layer in self.histogram_layers['mhsa']:
                    output_size = int(self.original_embedding_dim / layer.bin_widths_conv.out_channels)
                    layer.hist_pool = nn.AdaptiveAvgPool1d(output_size)
                print(f"Added {num_layers} non-shared MHSA histogram layers with {num_bins} bins and output size {output_size}")
            if 'ffn' in histogram_location:
                self.histogram_layers['ffn'] = nn.ModuleList([HistogramLayer(in_channels=self.original_embedding_dim,
                                                                             kernel_size=1, dim=1,
                                                                             num_bins=num_bins, stride=1,
                                                                             normalize_count=True, normalize_bins=True) 
                                                              for _ in range(num_layers)])
                for layer in self.histogram_layers['ffn']:
                    output_size = int(self.original_embedding_dim / layer.bin_widths_conv.out_channels)
                    layer.hist_pool = nn.AdaptiveAvgPool1d(output_size)
                print(f"Added {num_layers} non-shared FFN histogram layers with {num_bins} bins and output size {output_size}")
        
        self.classifier = nn.Linear(self.original_embedding_dim, num_labels)
        self.freeze_base_model()

    def freeze_base_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for hist_layer in self.histogram_layers.values():
            for param in hist_layer.parameters():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("Base model frozen. Only histogram layers and classifier are trainable.")
                
    def forward(self, input_values):
        hidden_states = self.model.embeddings(input_values)
        
        for i, layer in enumerate(self.model.encoder.layer):
            residual = hidden_states
            hidden_states = layer.layernorm_before(hidden_states)
            
            # MHSA sublayer
            attn_output = layer.attention(hidden_states)[0]
            if 'mhsa' in self.histogram_location:
                hist_layer = self.histogram_layers['mhsa'] if self.hist_shared else self.histogram_layers['mhsa'][i]
                hist_features = hist_layer(residual.permute(0, 2, 1)).permute(0, 2, 1)
                hist_features_flat = hist_features.reshape(residual.shape[0], -1)
                hist_features_flat = hist_features_flat.unsqueeze(1).expand(-1, residual.shape[1], -1)
                if self.histogram_mode == 'parallel':
                    attn_output = attn_output + hist_features_flat
                else:  # sequential
                    attn_output = hist_features_flat
            hidden_states = residual + attn_output
            
            # FFN sublayer
            residual = hidden_states
            hidden_states = layer.layernorm_after(hidden_states)
            ffn_output = layer.intermediate(hidden_states)
            ffn_output = layer.output(ffn_output, hidden_states)
            
            if 'ffn' in self.histogram_location:
                hist_layer = self.histogram_layers['ffn'] if self.hist_shared else self.histogram_layers['ffn'][i]
                hist_features = hist_layer(residual.permute(0, 2, 1)).permute(0, 2, 1)
                hist_features_flat = hist_features.reshape(residual.shape[0], -1)
                hist_features_flat = hist_features_flat.unsqueeze(1).expand(-1, residual.shape[1], -1)
                if self.histogram_mode == 'parallel':
                    ffn_output = ffn_output + hist_features_flat
                else:  # sequential
                    ffn_output = hist_features_flat
            
            hidden_states = residual + ffn_output
        
        hidden_states = self.model.layernorm(hidden_states)
        logits = self.classifier(hidden_states[:, 0])
        return logits   