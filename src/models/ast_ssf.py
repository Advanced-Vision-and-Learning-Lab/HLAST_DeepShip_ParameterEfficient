import torch
import torch.nn as nn
import os
import wget
os.environ['TORCH_HOME'] = 'pretrained_models'
import timm
from timm.models.layers import to_2tuple, trunc_normal_

###############################################################################
# SSF helper functions
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
    Supports both (B, N, C) and (B, C, H, W) inputs.
    """
    assert scale.shape == shift.shape, "Scale and shift must have the same shape."
    if x.shape[-1] == scale.shape[0]:
        # e.g. (B, N, C)
        return x * scale + shift
    elif x.shape[1] == scale.shape[0]:
        # e.g. (B, C, H, W)
        return x * scale.view(1, -1, 1, 1) + shift.view(1, -1, 1, 1)
    else:
        raise ValueError('Input shape does not match SSF scale/shift shape.')

class SSFParam(nn.Module):
    """
    A small module that holds a pair of learnable parameters for scaling & shifting.
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
# Minimal custom versions of Attention & Mlp that accept external SSF modules.
# We do NOT store SSF parameters inside these modules directly, to allow
# either shared or per-block SSF in the top-level ASTSSF constructor.
###############################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # qkv: in_features=dim, out_features=3*dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # final projection back to dim
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, ssf_qkv=None, ssf_proj=None):
        """
        x: (B, N, C)
        ssf_qkv, ssf_proj: callables that apply SSF if provided, else None.
        """
        B, N, C = x.shape

        # 1) qkv
        qkv_out = self.qkv(x)
        if ssf_qkv is not None:
            qkv_out = ssf_qkv(qkv_out)
        qkv_out = qkv_out.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv_out = qkv_out.permute(2, 0, 3, 1, 4)
        q, k, v = qkv_out.unbind(dim=0)  # each: (B, num_heads, N, C//num_heads)

        # 2) attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 3) attended output
        x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 4) final proj
        x_out = self.proj(x_out)
        if ssf_proj is not None:
            x_out = ssf_proj(x_out)
        x_out = self.proj_drop(x_out)

        return x_out

class Mlp(nn.Module):
    """
    A simple 2-layer MLP used in ViT blocks. 
    We accept optional external SSF callables for after fc1 and after fc2.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        # We define two dropouts with the same probability
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x, ssf_fc1=None, ssf_fc2=None):
        """
        x: (B, N, C)
        ssf_fc1, ssf_fc2: optional callables that do SSF (scale+shift).
        """
        x = self.fc1(x)
        if ssf_fc1 is not None:
            x = ssf_fc1(x)
        x = self.act(x)
        x = self.drop1(x)

        x = self.fc2(x)
        if ssf_fc2 is not None:
            x = ssf_fc2(x)
        x = self.drop2(x)

        return x

###############################################################################
# Override timm's PatchEmbed so we can handle non-224 inputs 
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
        # B,C,H,W -> B,C,HW -> B,HW,C
        x = x.flatten(2).transpose(1, 2)
        return x

###############################################################################
# AST model with SSF in up to 6 positions per block (3 MHSA, 3 FFN),
# with "shared vs. not shared" and "mhsa_only vs. full vs. single" options,
# adapted to DistilledVisionTransformer's Mlp structure.
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
        ssf_mode: 
          - "full"       => SSF in MHSA (3 positions) + FFN (3 positions)
          - "mhsa_only"  => SSF in MHSA only (3 positions)
          - "single"     => A single SSF applied after the first norm layer in MHSA
          
        ssf_shared:
          - If True, all SSF positions share one set of parameters across blocks for each location.
          - If False, each block has its own set.
        """
        super().__init__()
        assert timm.__version__ == '0.4.5', 'Use timm==0.4.5 for compatibility.'
        if verbose:
            print('--- AST Model with SSF (3, 6 or 1 positions per block) ---')
            print(f'ImageNet Pretrain: {imagenet_pretrain}, AudioSet Pretrain: {audioset_pretrain}')
            print(f'ssf_mode={ssf_mode}, ssf_shared={ssf_shared}')

        # Patch the timm ViT to use our custom PatchEmbed
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        self.ssf_mode = ssf_mode
        self.ssf_shared = ssf_shared

        # Create or load a timm ViT
        if not audioset_pretrain:
            # Standard timm code path...
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise ValueError('model_size must be one of [tiny224, small224, base224, base384].')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.oringal_hw = int(self.original_num_patches**0.5)

            # Adjust patch embedding for 1-channel input
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print(f'fstride={fstride}, tstride={tstride}, #patches={num_patches}')

            new_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16,16), stride=(fstride,tstride))
            if imagenet_pretrain:
                # sum the existing 3-ch kernel
                new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # Adjust pos_embed shape
            if imagenet_pretrain:
                new_pos_embed = self.resize_pos_embed(self.v.pos_embed[:, 2:, :], self.oringal_hw, f_dim, t_dim)
                self.v.pos_embed = nn.Parameter(torch.cat([
                    self.v.pos_embed[:, :2, :],  # keep class+dist tokens
                    new_pos_embed
                ], dim=1))
            else:
                new_pe = nn.Parameter(torch.zeros(1, num_patches + 2, self.original_embedding_dim))
                trunc_normal_(new_pe, std=.02)
                self.v.pos_embed = new_pe

        else:
            # AudioSet-pretrained path (only base384)
            if not imagenet_pretrain:
                raise ValueError('AudioSet-only pretrained not currently supported.')
            if model_size != 'base384':
                raise ValueError('Only base384 with AudioSet pretrained is supported here.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not os.path.exists('pretrained_models/audioset_10_10_0.4593.pth'):
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='pretrained_models/audioset_10_10_0.4593.pth')

            # First load a minimal instance
            audio_model = ASTSSF(label_dim=527, fstride=10, tstride=10,
                                 input_fdim=128, input_tdim=1024,
                                 imagenet_pretrain=False, audioset_pretrain=False,
                                 model_size='base384', verbose=False,
                                 ssf_shared=ssf_shared, ssf_mode=ssf_mode)
            audio_model = nn.DataParallel(audio_model)
            sd = torch.load('pretrained_models/audioset_10_10_0.4593.pth', map_location=device, weights_only=True)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print(f'fstride={fstride}, tstride={tstride}, #patches={num_patches}')

            new_pos_embed = self.resize_audioset_pe(self.v.pos_embed[:, 2:, :], f_dim, t_dim)
            self.v.pos_embed = nn.Parameter(torch.cat([
                self.v.pos_embed[:, :2, :],  # keep class+dist tokens
                new_pos_embed
            ], dim=1))

        # Replace the timm blocks with our custom Attn & Mlp so we can pass SSF.
        dim = self.original_embedding_dim
        for b_i, blk in enumerate(self.v.blocks):
            # Extract dropout param from old Mlp
            old_attn = blk.attn
            old_mlp = blk.mlp
            num_heads = old_attn.num_heads
            qkv_bias = (old_attn.qkv.bias is not None)
            attn_drop = old_attn.attn_drop.p
            proj_drop = old_attn.proj_drop.p

            # DistilledVisionTransformer's Mlp typically has a single dropout module .drop
            # We'll replicate that dropout value in drop1 & drop2 internally.
            if hasattr(old_mlp, 'drop'):
                mlp_drop = old_mlp.drop.p
            else:
                # fallback in case it's named differently
                mlp_drop = 0.0

            # Hidden dim from fc1
            mlp_hidden_dim = old_mlp.fc1.out_features

            # Create new custom submodules
            new_attn = Attention(dim, num_heads, qkv_bias, attn_drop, proj_drop)
            new_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                          out_features=dim, drop=mlp_drop)

            # Copy existing weights
            new_attn.qkv.weight.data.copy_(old_attn.qkv.weight.data)
            new_attn.qkv.bias.data.copy_(old_attn.qkv.bias.data)
            new_attn.proj.weight.data.copy_(old_attn.proj.weight.data)
            new_attn.proj.bias.data.copy_(old_attn.proj.bias.data)

            new_mlp.fc1.weight.data.copy_(old_mlp.fc1.weight.data)
            new_mlp.fc1.bias.data.copy_(old_mlp.fc1.bias.data)
            new_mlp.fc2.weight.data.copy_(old_mlp.fc2.weight.data)
            new_mlp.fc2.bias.data.copy_(old_mlp.fc2.bias.data)

            # Update references
            blk.attn = new_attn
            blk.mlp = new_mlp
            # Keep blk.norm1, blk.norm2, and blk.drop_path as-is.

        # Create the SSF parameters for each of the 6 (or 3 or 1) positions.
        num_blocks = len(self.v.blocks)

        # --- Create SSF parameters for MHSA ---
        # For MHSA, "full" and "mhsa_only" use 3 SSF layers,
        # while "single" uses only one SSF after norm1.
        if self.ssf_mode == "single":
            if ssf_shared:
                self.ssf_mhsa_norm_in = SSFParam(dim)
                self.ssf_mhsa_qkv = None
                self.ssf_mhsa_proj = None
            else:
                self.ssf_mhsa_norm_in = nn.ModuleList([SSFParam(dim) for _ in range(num_blocks)])
                self.ssf_mhsa_qkv = None
                self.ssf_mhsa_proj = None
        elif self.ssf_mode in ["full", "mhsa_only"]:
            if ssf_shared:
                self.ssf_mhsa_norm_in = SSFParam(dim)
                self.ssf_mhsa_qkv     = SSFParam(dim * 3)
                self.ssf_mhsa_proj    = SSFParam(dim)
            else:
                self.ssf_mhsa_norm_in = nn.ModuleList([SSFParam(dim) for _ in range(num_blocks)])
                self.ssf_mhsa_qkv     = nn.ModuleList([SSFParam(dim * 3) for _ in range(num_blocks)])
                self.ssf_mhsa_proj    = nn.ModuleList([SSFParam(dim) for _ in range(num_blocks)])
        else:
            raise ValueError("Invalid ssf_mode. Choose from ['full', 'mhsa_only', 'single'].")

        # --- Create SSF parameters for FFN ---
        # Only add FFN SSF if ssf_mode == "full"
        if self.ssf_mode == "full":
            if ssf_shared:
                self.ssf_ffn_norm_in = SSFParam(dim)
                hidden_dim = self.v.blocks[0].mlp.fc1.out_features
                self.ssf_ffn_fc1 = SSFParam(hidden_dim)
                self.ssf_ffn_fc2 = SSFParam(dim)
            else:
                self.ssf_ffn_norm_in = nn.ModuleList([SSFParam(dim) for _ in range(num_blocks)])
                hidden_dim = self.v.blocks[0].mlp.fc1.out_features
                self.ssf_ffn_fc1 = nn.ModuleList([SSFParam(hidden_dim) for _ in range(num_blocks)])
                self.ssf_ffn_fc2 = nn.ModuleList([SSFParam(dim) for _ in range(num_blocks)])

        # --- Create classifier head ---
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, label_dim)
        )

        # *** IMPORTANT: Now freeze all parameters in self.v ***
        self.freeze_base_model()
        # At this point, only self.ssf_* modules and mlp_head remain trainable.

    def freeze_base_model(self):
        """ Freeze all parameters in self.v (the base ViT) """
        for param in self.v.parameters():
            param.requires_grad = False
        print("Base model frozen. Only SSF parameters + mlp_head remain trainable.")

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        """
        Utility to figure out the # of patches in freq/time dims.
        """
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16,16), stride=(fstride,tstride))
        out = test_proj(test_input)
        return out.shape[2], out.shape[3]

    def resize_pos_embed(self, pos_embed, orig_hw, f_dim, t_dim):
        """
        Resizes the 'pos_embed' from the original (e.g. 14x14) to (f_dim x t_dim).
        pos_embed is shape (1, num_patches, C).
        We reshape to (1, C, 14, 14), do interpolation, reshape back.
        """
        pos_embed = pos_embed.reshape(1, orig_hw * orig_hw, -1).transpose(1,2)
        pos_embed = pos_embed.reshape(1, -1, orig_hw, orig_hw)
        # Now interpolate
        if t_dim <= orig_hw:
            pos_embed = pos_embed[:, :, :, (orig_hw//2 - t_dim//2):(orig_hw//2 - t_dim//2 + t_dim)]
        else:
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=(orig_hw, t_dim), mode='bilinear')
        if f_dim <= orig_hw:
            pos_embed = pos_embed[:, :, (orig_hw//2 - f_dim//2):(orig_hw//2 - f_dim//2 + f_dim), :]
        else:
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=(f_dim, t_dim), mode='bilinear')
        pos_embed = pos_embed.reshape(1, -1, f_dim*t_dim).transpose(1,2)
        return pos_embed

    def resize_audioset_pe(self, pos_embed, f_dim, t_dim):
        """
        The AudioSet pretrained model has pos_embed for a shape of 12 x 101 => 1212 patches.
        We reshape to (1, 768, 12, 101), do interpolation, reshape back.
        """
        pos_embed = pos_embed.reshape(1, 1212, 768).transpose(1,2).reshape(1, 768, 12, 101)
        # t_dim
        if t_dim < 101:
            pos_embed = pos_embed[:, :, :, 50 - t_dim//2 : 50 - t_dim//2 + t_dim]
        else:
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=(12, t_dim), mode='bilinear')
        # f_dim
        if f_dim < 12:
            pos_embed = pos_embed[:, :, 6 - f_dim//2 : 6 - f_dim//2 + f_dim, :]
        elif f_dim > 12:
            pos_embed = torch.nn.functional.interpolate(pos_embed, size=(f_dim, t_dim), mode='bilinear')
        # flatten
        return pos_embed.reshape(1, 768, f_dim*t_dim).transpose(1,2)

    def forward(self, x):
        B = x.shape[0]
        # 1) patch embedding
        x = self.v.patch_embed(x)

        # 2) add cls/dist tokens
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        # 3) add pos_embed, drop
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        # 4) run the blocks
        for i, blk in enumerate(self.v.blocks):
            # === MHSA sub-layer ===
            residual = x
            x_norm = blk.norm1(x)

            # (1) Apply SSF after norm1 based on mode
            if self.ssf_mode == "single":
                if self.ssf_shared:
                    x_norm = self.ssf_mhsa_norm_in(x_norm)
                else:
                    x_norm = self.ssf_mhsa_norm_in[i](x_norm)
                ssf_qkv = None
                ssf_proj = None
            else:  # "full" and "mhsa_only" modes: use all 3 SSF positions
                if self.ssf_shared:
                    x_norm = self.ssf_mhsa_norm_in(x_norm)
                    ssf_qkv  = self.ssf_mhsa_qkv
                    ssf_proj = self.ssf_mhsa_proj
                else:
                    x_norm = self.ssf_mhsa_norm_in[i](x_norm)
                    ssf_qkv  = self.ssf_mhsa_qkv[i]
                    ssf_proj = self.ssf_mhsa_proj[i]

            # (2) SSF after qkv & (3) after proj occur inside attention
            attn_out = blk.attn(x_norm, ssf_qkv=ssf_qkv, ssf_proj=ssf_proj)
            x = residual + attn_out
            x = blk.drop_path(x)

            # === FFN sub-layer ===
            residual = x
            x_norm = blk.norm2(x)
            if self.ssf_mode == "full":
                if self.ssf_shared:
                    x_norm = self.ssf_ffn_norm_in(x_norm)
                    ssf_fc1 = self.ssf_ffn_fc1
                    ssf_fc2 = self.ssf_ffn_fc2
                else:
                    x_norm = self.ssf_ffn_norm_in[i](x_norm)
                    ssf_fc1 = self.ssf_ffn_fc1[i]
                    ssf_fc2 = self.ssf_ffn_fc2[i]
            else:
                ssf_fc1 = None
                ssf_fc2 = None

            # (5) SSF after fc1 & (6) after fc2 occur inside Mlp
            ffn_out = blk.mlp(
                x_norm,
                ssf_fc1=ssf_fc1,
                ssf_fc2=ssf_fc2
            )
            x = residual + ffn_out
            x = blk.drop_path(x)

        # 5) average the cls/dist tokens
        x = (x[:, 0] + x[:, 1]) / 2

        # 6) final classifier
        x = self.mlp_head(x)

        return x
