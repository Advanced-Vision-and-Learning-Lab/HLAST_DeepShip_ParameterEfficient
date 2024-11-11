import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import os
import wget
os.environ['TORCH_HOME'] = 'pretrained_models'
import timm
from timm.models.layers import to_2tuple,trunc_normal_

import pdb

from .RBFHistogramPooling import HistogramLayer


# override the timm package to relax the input shape constraint.
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class ASTHistogram(nn.Module):
    """
    The AST model.
    :param label_dim: the label dimension, i.e., the number of total classes, it is 527 for AudioSet, 50 for ESC-50, and 35 for speechcommands v2-35
    :param fstride: the stride of patch spliting on the frequency dimension, for 16*16 patchs, fstride=16 means no overlap, fstride=10 means overlap of 6
    :param tstride: the stride of patch spliting on the time dimension, for 16*16 patchs, tstride=16 means no overlap, tstride=10 means overlap of 6
    :param input_fdim: the number of frequency bins of the input spectrogram
    :param input_tdim: the number of time frames of the input spectrogram
    :param imagenet_pretrain: if use ImageNet pretrained model
    :param audioset_pretrain: if use full AudioSet and ImageNet pretrained model
    :param model_size: the model size of AST, should be in [tiny224, small224, base224, base384], base224 and base 384 are same model, but are trained differently during ImageNet pretraining.
    """
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 imagenet_pretrain=True, model_size='base384', verbose=True, audioset_pretrain=True, hist_shared=True,
                 NumBins=16, histogram_mode='parallel', histogram_location='ffn'):

        super(ASTHistogram, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose == True:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain),str(audioset_pretrain)))
        # override timm input shape restriction
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # if AudioSet pretraining is not used (but ImageNet pretraining may still apply)
        if audioset_pretrain == False:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise Exception('Model size must be one of tiny224, small224, base224, base384.')
            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches ** 0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            # automatcially get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain == True:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain == True:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token), reshape it to original 2D shape (24*24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                # TODO can use sinusoidal positional embedding instead
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('currently model pretrained on only audioset is not supported, please set imagenet_pretrain = True to use audioset pretrained model.')
            if model_size != 'base384':
                raise ValueError('currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if os.path.exists('pretrained_models/audioset_10_10_0.4593.pth') == False:
                # this model performs 0.4593 mAP on the audioset eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('pretrained_models/audioset_10_10_0.4593.pth', map_location=device)
            audio_model = ASTHistogram(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]       
            
            print(f"\nNumber of transformer blocks: {len(self.v.blocks)}")
    
            self.histogram_mode = histogram_mode
            self.histogram_location = histogram_location
            
            if hist_shared:
                # shared weights:

                self.histogram_layers_mhsa = None
                self.histogram_layers_ffn = None
                self.histogram_layers_out = None
                
                if self.histogram_location in ['all', 'mhsa_ffn','mhsa_out', 'mhsa']:
                    histogram_layer_mhsa = HistogramLayer(in_channels = 768,
                                                      kernel_size = 1, dim=1,
                                                      num_bins=NumBins, stride=1,
                                                      normalize_count=True, normalize_bins=True)
                    
                    self.histogram_layers_mhsa = nn.ModuleList([
                        histogram_layer_mhsa for _ in range(len(self.v.blocks))
                    ])
                    output_size = int(self.original_embedding_dim / histogram_layer_mhsa.bin_widths_conv.out_channels)
                    for layer in self.histogram_layers_mhsa:
                        layer.hist_pool = nn.AdaptiveAvgPool1d(output_size)
                    print(f"Histogram Layers for MHSA Initialized: {self.histogram_layers_mhsa}\n")
                    
                if self.histogram_location in ['all', 'mhsa_ffn', 'ffn_out', 'ffn']:
                    histogram_layer_ffn = HistogramLayer(in_channels = 768,
                                                      kernel_size = 1, dim=1,
                                                      num_bins=NumBins, stride=1,
                                                      normalize_count=True, normalize_bins=True)
                    self.histogram_layers_ffn = nn.ModuleList([
                        histogram_layer_ffn for _ in range(len(self.v.blocks))
                    ])
                    output_size = int(self.original_embedding_dim / histogram_layer_ffn.bin_widths_conv.out_channels)
                    for layer in self.histogram_layers_ffn:
                        layer.hist_pool = nn.AdaptiveAvgPool1d(output_size)
                    print(f"Histogram Layers for FFN Initialized: {self.histogram_layers_ffn}\n")
                             
                if self.histogram_location in ['all', 'mhsa_out', 'ffn_out', 'out']:
                    histogram_layer_out = HistogramLayer(in_channels = 768,
                                                      kernel_size = 1, dim=1,
                                                      num_bins=NumBins, stride=1,
                                                      normalize_count=True, normalize_bins=True)
                    self.histogram_layers_out = nn.ModuleList([
                        histogram_layer_out for _ in range(len(self.v.blocks))
                    ])
                    output_size = int(self.original_embedding_dim / histogram_layer_out.bin_widths_conv.out_channels)
                    for layer in self.histogram_layers_out:
                        layer.hist_pool = nn.AdaptiveAvgPool1d(output_size)
                    print(f"Histogram Layers for OUTPUT Initialized: {self.histogram_layers_out}\n")
                    
                        
            elif not hist_shared:
            # histogram layer disticnt weights:
                
                if self.histogram_location in ['all', 'mhsa_ffn', 'mhsa_out', 'mhsa']:
                    self.histogram_layers_mhsa = nn.ModuleList([
                        HistogramLayer(in_channels=768,
                                        kernel_size=1, dim=1,
                                        num_bins=NumBins, stride=1,
                                        normalize_count=True, normalize_bins=True) 
                        for _ in range(len(self.v.blocks))  # Create a new instance for each block
                    ])
                    output_size = int(self.original_embedding_dim / self.histogram_layers_mhsa[0].bin_widths_conv.out_channels)
        
                    for layer in self.histogram_layers_mhsa:
                        layer.hist_pool = nn.AdaptiveAvgPool1d(output_size)
                    print(f"Histogram Layers for MHSA Initialized: {self.histogram_layers_mhsa}\n")
                
                if self.histogram_location in ['all', 'mhsa_ffn', 'ffn_out', 'ffn']:
                    self.histogram_layers_ffn = nn.ModuleList([
                        HistogramLayer(in_channels=768,
                                        kernel_size=1, dim=1,
                                        num_bins=NumBins, stride=1,
                                        normalize_count=True, normalize_bins=True) 
                        for _ in range(len(self.v.blocks))  # Create a new instance for each block
                    ])
                    output_size = int(self.original_embedding_dim / self.histogram_layers_ffn[0].bin_widths_conv.out_channels)
                    for layer in self.histogram_layers_ffn:
                        layer.hist_pool = nn.AdaptiveAvgPool1d(output_size)
                    print(f"Histogram Layers for FFN Initialized: {self.histogram_layers_ffn}\n")
                
                if self.histogram_location in ['all', 'mhsa_out', 'ffn_out', 'out']:
                    self.histogram_layers_out = nn.ModuleList([
                        HistogramLayer(in_channels=768,
                                        kernel_size=1, dim=1,
                                        num_bins=NumBins, stride=1,
                                        normalize_count=True, normalize_bins=True) 
                        for _ in range(len(self.v.blocks))  # Create a new instance for each block
                    ])
                    output_size = int(self.original_embedding_dim / self.histogram_layers_out[0].bin_widths_conv.out_channels)
                    for layer in self.histogram_layers_out:
                        layer.hist_pool = nn.AdaptiveAvgPool1d(output_size)
                    print(f"Histogram Layers for OUTPUT Initialized: {self.histogram_layers_out}\n")


            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )
            
    
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose == True:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))
    
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original audioset (10s), then cut the positional embedding
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            # otherwise interpolate
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            # otherwise interpolate
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            
            self.freeze_base_model()
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim
    
    def freeze_base_model(self):
        # Freeze base model parameters
        for param in self.v.parameters():
            param.requires_grad = False
        
        # Unfreeze histogram layers if they exist and are not None
        if hasattr(self, 'histogram_layers_mhsa') and self.histogram_layers_mhsa is not None:
            for layer in self.histogram_layers_mhsa:
                for param in layer.parameters():
                    param.requires_grad = True
        
        if hasattr(self, 'histogram_layers_ffn') and self.histogram_layers_ffn is not None:
            for layer in self.histogram_layers_ffn:
                for param in layer.parameters():
                    param.requires_grad = True
        
        if hasattr(self, 'histogram_layers_out') and self.histogram_layers_out is not None:
            for layer in self.histogram_layers_out:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Unfreeze MLP head
        for param in self.mlp_head.parameters():
            param.requires_grad = True
        
        print("Base model frozen. Only initialized histogram layers and classifier are trainable.")
        

    @autocast()
    def forward(self, x):
        B = x.shape[0]

        x = self.v.patch_embed(x)

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for i, blk in enumerate(self.v.blocks):
            # MHSA sublayer
            residual = x
            x = blk.norm1(x)

            if self.histogram_location in ['all', 'mhsa_ffn', 'mhsa_out', 'mhsa']:
                if self.histogram_mode == 'parallel':
                    attn_out = blk.attn(x)
                    hist_features = self.histogram_layers_mhsa[i](x.permute(0, 2, 1)).permute(0, 2, 1)
                    hist_features_flat = hist_features.reshape(B, -1)
                    hist_features_flat = hist_features_flat.unsqueeze(1).expand(-1, x.shape[1], -1)
                    attn_out = attn_out + hist_features_flat 

                elif self.histogram_mode == 'sequential':
                    attn_out = blk.attn(x)
                    hist_features = self.histogram_layers_mhsa[i](attn_out.permute(0, 2, 1)).permute(0, 2, 1)
                    hist_features_flat = hist_features.reshape(B, -1)
                    hist_features_flat = hist_features_flat.unsqueeze(1).expand(-1, x.shape[1], -1)
                    attn_out = hist_features_flat  # Sequential applies after attention

            else:
                attn_out = blk.attn(x)

            x = attn_out + residual
            x = blk.drop_path(x)


            # FFN sublayer
            residual = x
            x = blk.norm2(x)


            if self.histogram_location in ['all', 'mhsa_ffn', 'ffn_out', 'ffn']:
                if self.histogram_mode == 'parallel':
                    ffn_out = blk.mlp(x)
                    hist_features = self.histogram_layers_ffn[i](x.permute(0, 2, 1)).permute(0, 2, 1)
                    hist_features_flat = hist_features.reshape(B, -1)
                    hist_features_flat = hist_features_flat.unsqueeze(1).expand(-1, x.shape[1], -1)
                    ffn_out = ffn_out + hist_features_flat

                elif self.histogram_mode == 'sequential':
                    ffn_out = blk.mlp(x)
                    hist_features = self.histogram_layers_ffn[i](ffn_out.permute(0, 2, 1)).permute(0, 2, 1)
                    hist_features_flat = hist_features.reshape(B, -1)
                    hist_features_flat = hist_features_flat.unsqueeze(1).expand(-1, x.shape[1], -1)
                    ffn_out = hist_features_flat  # Sequential applies after ffn
    
            else:
                ffn_out = blk.mlp(x)
            
            x = ffn_out + residual
            x = blk.drop_path(x)


            if self.histogram_location in ['all', 'mhsa_out', 'ffn_out', 'out']:
                if self.histogram_mode == 'parallel':
                    hist_features = self.histogram_layers_out[i](x.permute(0, 2, 1)).permute(0, 2, 1)
                    hist_features_flat = hist_features.reshape(B, -1)
                    hist_features_flat = hist_features_flat.unsqueeze(1).expand(-1, x.shape[1], -1)
                    x = x + hist_features_flat

        x = (x[:, 0] + x[:, 1]) / 2
        x = self.mlp_head(x)

        return x 

# import torch
# import torch.nn as nn
# from transformers import ASTModel
# from .RBFHistogramPooling import HistogramLayer

# class ASTHistogram(nn.Module):
#     def __init__(self, num_labels, max_length, num_mel_bins, num_bins,
#                  histogram_location='mhsa_ffn', hist_shared=True, histogram_mode='parallel', 
#                  model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
#         super().__init__()
        
#         self.model = ASTModel.from_pretrained(model_ckpt, max_length=max_length, num_mel_bins=num_mel_bins, ignore_mismatched_sizes=True)
#         self.model_config = self.model.config
#         self.original_embedding_dim = self.model_config.hidden_size
#         self.histogram_location = histogram_location
#         self.histogram_mode = histogram_mode
#         self.hist_shared = hist_shared
#         self.num_bins = num_bins
        
#         self.histogram_layers = nn.ModuleDict()
#         num_layers = len(self.model.encoder.layer)
        
#         print(f"Initializing ASTHistogram with the following configuration:")
#         print(f"Histogram location: {histogram_location}")
#         print(f"Histogram mode: {histogram_mode}")
#         print(f"Histogram shared: {hist_shared}")
#         print(f"Number of bins: {self.num_bins}")
#         print(f"Number of transformer layers: {num_layers}")
        
#         if hist_shared:
#             if 'mhsa' in histogram_location:
#                 self.histogram_layers['mhsa'] = HistogramLayer(in_channels=self.original_embedding_dim,
#                                                                kernel_size=1, dim=1,
#                                                                num_bins=num_bins, stride=1,
#                                                                normalize_count=True, normalize_bins=True)
#                 output_size = int(self.original_embedding_dim / self.histogram_layers['mhsa'].bin_widths_conv.out_channels)
#                 self.histogram_layers['mhsa'].hist_pool = nn.AdaptiveAvgPool1d(output_size)
#                 print(f"Added shared MHSA histogram layer with {num_bins} bins and output size {output_size}")
#             if 'ffn' in histogram_location:
#                 self.histogram_layers['ffn'] = HistogramLayer(in_channels=self.original_embedding_dim,
#                                                               kernel_size=1, dim=1,
#                                                               num_bins=num_bins, stride=1,
#                                                               normalize_count=True, normalize_bins=True)
#                 output_size = int(self.original_embedding_dim / self.histogram_layers['ffn'].bin_widths_conv.out_channels)
#                 self.histogram_layers['ffn'].hist_pool = nn.AdaptiveAvgPool1d(output_size)
#                 print(f"Added shared FFN histogram layer with {num_bins} bins and output size {output_size}")
#         else:
#             if 'mhsa' in histogram_location:
#                 self.histogram_layers['mhsa'] = nn.ModuleList([HistogramLayer(in_channels=self.original_embedding_dim,
#                                                                               kernel_size=1, dim=1,
#                                                                               num_bins=num_bins, stride=1,
#                                                                               normalize_count=True, normalize_bins=True) 
#                                                                for _ in range(num_layers)])
#                 for layer in self.histogram_layers['mhsa']:
#                     output_size = int(self.original_embedding_dim / layer.bin_widths_conv.out_channels)
#                     layer.hist_pool = nn.AdaptiveAvgPool1d(output_size)
#                 print(f"Added {num_layers} non-shared MHSA histogram layers with {num_bins} bins and output size {output_size}")
#             if 'ffn' in histogram_location:
#                 self.histogram_layers['ffn'] = nn.ModuleList([HistogramLayer(in_channels=self.original_embedding_dim,
#                                                                              kernel_size=1, dim=1,
#                                                                              num_bins=num_bins, stride=1,
#                                                                              normalize_count=True, normalize_bins=True) 
#                                                               for _ in range(num_layers)])
#                 for layer in self.histogram_layers['ffn']:
#                     output_size = int(self.original_embedding_dim / layer.bin_widths_conv.out_channels)
#                     layer.hist_pool = nn.AdaptiveAvgPool1d(output_size)
#                 print(f"Added {num_layers} non-shared FFN histogram layers with {num_bins} bins and output size {output_size}")
        
#         self.classifier = nn.Linear(self.original_embedding_dim, num_labels)
#         self.freeze_base_model()

#     def freeze_base_model(self):
#         for param in self.model.parameters():
#             param.requires_grad = False
#         for hist_layer in self.histogram_layers.values():
#             for param in hist_layer.parameters():
#                 param.requires_grad = True
#         for param in self.classifier.parameters():
#             param.requires_grad = True
#         print("Base model frozen. Only histogram layers and classifier are trainable.")
                
#     def forward(self, input_values):
#         hidden_states = self.model.embeddings(input_values)
        
#         for i, layer in enumerate(self.model.encoder.layer):
#             residual = hidden_states
#             hidden_states = layer.layernorm_before(hidden_states)
            
#             # MHSA sublayer
#             attn_output = layer.attention(hidden_states)[0]
#             if 'mhsa' in self.histogram_location:
#                 hist_layer = self.histogram_layers['mhsa'] if self.hist_shared else self.histogram_layers['mhsa'][i]
#                 hist_features = hist_layer(residual.permute(0, 2, 1)).permute(0, 2, 1)
#                 hist_features_flat = hist_features.reshape(residual.shape[0], -1)
#                 hist_features_flat = hist_features_flat.unsqueeze(1).expand(-1, residual.shape[1], -1)
#                 if self.histogram_mode == 'parallel':
#                     attn_output = attn_output + hist_features_flat
#                 else:  # sequential
#                     attn_output = hist_features_flat
#             hidden_states = residual + attn_output
            
#             # FFN sublayer
#             residual = hidden_states
#             hidden_states = layer.layernorm_after(hidden_states)
#             ffn_output = layer.intermediate(hidden_states)
#             ffn_output = layer.output(ffn_output, hidden_states)
            
#             if 'ffn' in self.histogram_location:
#                 hist_layer = self.histogram_layers['ffn'] if self.hist_shared else self.histogram_layers['ffn'][i]
#                 hist_features = hist_layer(residual.permute(0, 2, 1)).permute(0, 2, 1)
#                 hist_features_flat = hist_features.reshape(residual.shape[0], -1)
#                 hist_features_flat = hist_features_flat.unsqueeze(1).expand(-1, residual.shape[1], -1)
#                 if self.histogram_mode == 'parallel':
#                     ffn_output = ffn_output + hist_features_flat
#                 else:  # sequential
#                     ffn_output = hist_features_flat
            
#             hidden_states = residual + ffn_output
        
#         hidden_states = self.model.layernorm(hidden_states)
#         logits = self.classifier(hidden_states[:, 0])
#         return logits   