import torch
import torch.nn as nn
import os
import wget
os.environ['TORCH_HOME'] = 'pretrained_models'
import timm
from timm.models.layers import to_2tuple, trunc_normal_

###############################################################################
# SSF helper functions and module
###############################################################################

def init_ssf_scale_shift(dim):
    scale = nn.Parameter(torch.ones(dim))
    shift = nn.Parameter(torch.zeros(dim))
    nn.init.normal_(scale, mean=1, std=0.02)
    nn.init.normal_(shift, std=0.02)
    return scale, shift

def ssf_ada(x, scale, shift):
    """
    Apply an affine transformation with learnable scale and shift.
    This function supports both (B, N, C) and (B, C, H, W) inputs.
    """
    assert scale.shape == shift.shape, "Scale and shift must have the same shape."
    if x.shape[-1] == scale.shape[0]:
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('Input tensor shape does not match the shape of the scale factor.')

class SSFParam(nn.Module):
    """
    A small module that holds a pair of learnable parameters for scaling and shifting.
    """
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        nn.init.normal_(self.scale, mean=1, std=0.02)
        nn.init.normal_(self.shift, std=0.02)
    def forward(self, x):
        return ssf_ada(x, self.scale, self.shift)

###############################################################################
# Override timm's PatchEmbed to relax input shape constraint
###############################################################################

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

###############################################################################
# AST model with SSF integrated in place of adapters.
#
# New constructor arguments:
#   - ssf_shared (bool): whether to share the SSF parameters across blocks (default: True).
#   - ssf_mode (str): either "full" (apply SSF in both MHSA and FFN branches) or "mhsa_only"
#       (apply SSF only in the MHSA branch). The default "full" mode applies SSF both
#       before and after each sub-layer (i.e. after norm1 & after attention, and after norm2 & after FFN).
###############################################################################

class ASTSSF(nn.Module):
    def __init__(self, 
                 label_dim=527, 
                 fstride=10, 
                 tstride=10, 
                 input_fdim=128, 
                 input_tdim=1024, 
                 imagenet_pretrain=True, 
                 audioset_pretrain=True, 
                 model_size='base384', 
                 verbose=True,
                 ssf_shared=True,
                 ssf_mode='full'):
        """
        Note: This version replaces the adapter integration with SSF.
        ssf_mode: "full" applies SSF before & after both MHSA and FFN.
                  "mhsa_only" applies SSF only for the MHSA branch.
        """
        super(ASTSSF, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose:
            print('---------------AST Model Summary (with SSF) ---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain), str(audioset_pretrain)))
        # Override timm input shape restriction.
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        # If AudioSet pretraining is not used (but ImageNet pretraining may still apply)
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

            # Automatically get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # The linear projection layer for 1-channel input
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # Positional embedding modification
            if imagenet_pretrain:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1, 2)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # Now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain and not imagenet_pretrain:
                raise ValueError('Currently model pretrained on only AudioSet is not supported. Please set imagenet_pretrain=True.')
            if model_size != 'base384':
                raise ValueError('Currently only base384 AudioSet pretrained model is supported.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not os.path.exists('pretrained_models/audioset_10_10_0.4593.pth'):
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('pretrained_models/audioset_10_10_0.4593.pth', map_location=device, weights_only=True)
            # To load the AudioSet-pretrained model, 
            audio_model = ASTSSF(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                                     imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            
            print(f"\nNumber of transformer blocks: {len(self.v.blocks)}")
            
            # SSF configuration: remove adapter parameters and instead set up SSF.
            self.ssf_shared = ssf_shared
            self.ssf_mode = ssf_mode  # "full" or "mhsa_only"
            num_blocks = len(self.v.blocks)

            if self.ssf_shared:
                # A single shared SSF pair for each location.
                self.ssf_mhsa_in = SSFParam(self.original_embedding_dim)
                self.ssf_mhsa_out = SSFParam(self.original_embedding_dim)
                if self.ssf_mode == "full":
                    self.ssf_ffn_in = SSFParam(self.original_embedding_dim)
                    self.ssf_ffn_out = SSFParam(self.original_embedding_dim)
            else:
                # Distinct SSF parameters for each transformer block.
                self.ssf_mhsa_in = nn.ModuleList([SSFParam(self.original_embedding_dim) for _ in range(num_blocks)])
                self.ssf_mhsa_out = nn.ModuleList([SSFParam(self.original_embedding_dim) for _ in range(num_blocks)])
                if self.ssf_mode == "full":
                    self.ssf_ffn_in = nn.ModuleList([SSFParam(self.original_embedding_dim) for _ in range(num_blocks)])
                    self.ssf_ffn_out = nn.ModuleList([SSFParam(self.original_embedding_dim) for _ in range(num_blocks)])

            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )
            
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))
    
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - int(t_dim/2): 50 - int(t_dim/2) + t_dim]
            else:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - int(f_dim/2): 6 - int(f_dim/2) + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
            new_pos_embed = new_pos_embed.reshape(1, 768, num_patches).transpose(1, 2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

            # Freeze the base model parameters (they will not be updated)
            self.freeze_base_model()
    
    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def freeze_base_model(self):
        for param in self.v.parameters():
            param.requires_grad = False
        for param in self.mlp_head.parameters():
            param.requires_grad = True
        print("Base model frozen. Only SSF parameters and classifier head are trainable.")
    
    def forward(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        # Loop over transformer blocks and insert SSF
        for i, blk in enumerate(self.v.blocks):
            # ---- MHSA Sublayer ----
            residual = x
            x_norm = blk.norm1(x)
            # Apply SSF before MHSA (on norm1 output)
            if hasattr(self, 'ssf_mhsa_in'):
                if self.ssf_shared:
                    x_norm = self.ssf_mhsa_in(x_norm)
                else:
                    x_norm = self.ssf_mhsa_in[i](x_norm)
            attn_out = blk.attn(x_norm)
            # Apply SSF after MHSA
            if hasattr(self, 'ssf_mhsa_out'):
                if self.ssf_shared:
                    attn_out = self.ssf_mhsa_out(attn_out)
                else:
                    attn_out = self.ssf_mhsa_out[i](attn_out)
            x = residual + attn_out
            x = blk.drop_path(x)

            # ---- FFN Sublayer ----
            residual = x
            x_norm = blk.norm2(x)
            if self.ssf_mode == "full":
                if hasattr(self, 'ssf_ffn_in'):
                    if self.ssf_shared:
                        x_norm = self.ssf_ffn_in(x_norm)
                    else:
                        x_norm = self.ssf_ffn_in[i](x_norm)
                ffn_out = blk.mlp(x_norm)
                if hasattr(self, 'ssf_ffn_out'):
                    if self.ssf_shared:
                        ffn_out = self.ssf_ffn_out(ffn_out)
                    else:
                        ffn_out = self.ssf_ffn_out[i](ffn_out)
            else:  # ssf_mode == "mhsa_only": do not apply SSF in FFN branch
                ffn_out = blk.mlp(x_norm)
            x = residual + ffn_out
            x = blk.drop_path(x)

        # Pooling (average the cls and distillation tokens)
        x = (x[:, 0] + x[:, 1]) / 2
        x = self.mlp_head(x)
        return x
