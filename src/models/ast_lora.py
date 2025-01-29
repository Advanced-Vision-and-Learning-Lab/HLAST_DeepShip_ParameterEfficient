import torch
import torch.nn as nn
import os
import wget
os.environ['TORCH_HOME'] = 'pretrained_models'
import timm
from timm.models.layers import to_2tuple, trunc_normal_

import torch.nn.functional as F

class LoRAQKVAttention(nn.Module):
    """
    A single wrapper around Timm's MHSA to insert LoRA offsets into Q (and optionally V).
    The big attention weights remain in self.timm_attn (frozen).
    We only train the 'down_q', 'up_q', etc.
    """

    def __init__(self, 
                 timm_attn: nn.Module,
                 dim: int,
                 lora_target='q',   # 'q' or 'qv'
                 lora_rank=4,
                 lora_alpha=1.0):
        super().__init__()
        self.timm_attn = timm_attn
        self.dim        = dim
        self.lora_target = lora_target
        self.lora_rank   = lora_rank
        self.lora_alpha  = lora_alpha

        # Freeze Timm's big attn weights
        for p in self.timm_attn.parameters():
            p.requires_grad = False

        # LoRA for Q
        self.down_q = nn.Linear(dim, lora_rank, bias=False)
        self.up_q   = nn.Linear(lora_rank, dim, bias=False)
        nn.init.zeros_(self.down_q.weight)
        nn.init.zeros_(self.up_q.weight)

        # LoRA for V if we do 'qv'
        if self.lora_target == 'qv':
            self.down_v = nn.Linear(dim, lora_rank, bias=False)
            self.up_v   = nn.Linear(lora_rank, dim, bias=False)
            nn.init.zeros_(self.down_v.weight)
            nn.init.zeros_(self.up_v.weight)
        else:
            self.down_v = None
            self.up_v   = None

    def forward(self, x):
        B, N, C = x.shape
        # 1) Original Q, K, V from Timm (frozen)
        qkv = F.linear(x, self.timm_attn.qkv.weight, self.timm_attn.qkv.bias)
        qkv = qkv.reshape(B, N, 3, C).permute(2, 0, 1, 3)  # [3, B, N, C]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 2) Add LoRA offset to Q (and V if 'qv')
        q_offset = self.up_q(self.down_q(x)) * self.lora_alpha
        q = q + q_offset
        if self.lora_target == 'qv':
            v_offset = self.up_v(self.down_v(x)) * self.lora_alpha
            v = v + v_offset

        # 3) Standard multi-head attention
        num_heads = self.timm_attn.num_heads
        head_dim  = C // num_heads
        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        attn = (q * self.timm_attn.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.timm_attn.attn_drop(attn)

        x_out = attn @ v
        x_out = x_out.permute(0, 2, 1, 3).reshape(B, N, C)

        # output projection
        x_out = self.timm_attn.proj(x_out)
        x_out = self.timm_attn.proj_drop(x_out)
        return x_out


class PatchEmbed(nn.Module):
    """Identical to your original patch embedding override."""
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



class ASTLoRA(nn.Module):

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
                 lora_target='q',   # 'q' or 'qv'
                 lora_rank=4,
                 lora_alpha=1.0,
                 lora_shared=False):
        super().__init__()
        assert timm.__version__ == '0.4.5', 'Use timm==0.4.5 for compatibility.'

        if verbose:
            print('---------------AST Model Summary (LoRA)---------------')
            print(f'ImageNet pretraining: {imagenet_pretrain}, AudioSet pretraining: {audioset_pretrain}')

        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        self.lora_target = lora_target
        self.lora_rank   = lora_rank
        self.lora_alpha  = lora_alpha
        self.lora_shared = lora_shared

        if not audioset_pretrain:
            # ImageNet only path
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise ValueError("model_size must be one of [tiny224, small224, base224, base384]")

            self.original_num_patches = self.v.patch_embed.num_patches
            self.oringal_hw = int(self.original_num_patches**0.5)
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print(f'frequency stride={fstride}, time stride={tstride}')
                print(f'number of patches={num_patches}')

            # set up for 1-channel input
            new_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16,16), stride=(fstride,tstride))
            if imagenet_pretrain:
                new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias   = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if imagenet_pretrain:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(
                    1, self.original_num_patches, self.original_embedding_dim
                ).transpose(1,2).reshape(
                    1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw
                )
                # time dimension
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :, :, :, 
                        self.oringal_hw//2 - t_dim//2 : self.oringal_hw//2 - t_dim//2 + t_dim
                    ]
                else:
                    new_pos_embed = nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # freq dimension
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[
                        :, :, 
                        self.oringal_hw//2 - f_dim//2 : self.oringal_hw//2 - f_dim//2 + f_dim, 
                        :
                    ]
                else:
                    new_pos_embed = nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                new_pos_embed = nn.Parameter(torch.zeros(
                    1, self.v.patch_embed.num_patches+2, self.original_embedding_dim
                ))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

            self.build_lora_modules()
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )
            self.freeze_base_model()

        else:
            # AudioSet + ImageNet
            if not imagenet_pretrain:
                raise ValueError("AudioSet-only pretrained not supported.")
            if model_size != 'base384':
                raise ValueError("Only base384 AudioSet pretrained model is available.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if not os.path.exists('pretrained_models/audioset_10_10_0.4593.pth'):
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='pretrained_models/audioset_10_10_0.4593.pth')

            sd = torch.load('pretrained_models/audioset_10_10_0.4593.pth', map_location=device, weights_only=True)

            tmp = ASTLoRA(label_dim=527,
                          fstride=10, tstride=10,
                          input_fdim=128, input_tdim=1024,
                          imagenet_pretrain=False, audioset_pretrain=False,
                          model_size='base384', verbose=False,
                          lora_target=self.lora_target,
                          lora_rank=self.lora_rank,
                          lora_alpha=self.lora_alpha,
                          lora_shared=self.lora_shared)
            tmp = nn.DataParallel(tmp)
            tmp.load_state_dict(sd, strict=False)

            self.v = tmp.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]

            print(f"\nNumber of transformer blocks: {len(self.v.blocks)}")

            self.build_lora_modules()

            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print(f'frequency stride={fstride}, time stride={tstride}')
                print(f'number of patches={num_patches}')

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1,2).reshape(1,768,12,101)
            if t_dim < 101:
                new_pos_embed = new_pos_embed[:, :, :, 50 - t_dim//2 : 50 - t_dim//2 + t_dim]
            else:
                new_pos_embed = nn.functional.interpolate(new_pos_embed, size=(12, t_dim), mode='bilinear')
            if f_dim < 12:
                new_pos_embed = new_pos_embed[:, :, 6 - f_dim//2 : 6 - f_dim//2 + f_dim, :]
            elif f_dim > 12:
                new_pos_embed = nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')

            new_pos_embed = new_pos_embed.reshape(1,768, num_patches).transpose(1,2)
            self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))

            self.freeze_base_model()

    def build_lora_modules(self):
        """
        Replace each block's Timm Attention with LoRAQKVAttention once.
        If lora_shared=True, reuse one LoRAQKVAttention for all blocks.
        Otherwise, each block gets a distinct wrapper.
        """
        num_blocks = len(self.v.blocks)

        if self.lora_shared:
            # Build a single LoRAQKVAttention from block 0
            base_attn = self.v.blocks[0].attn
            if not isinstance(base_attn, LoRAQKVAttention):
                shared_lora = LoRAQKVAttention(
                    timm_attn=base_attn,
                    dim=self.original_embedding_dim,
                    lora_target=self.lora_target,
                    lora_rank=self.lora_rank,
                    lora_alpha=self.lora_alpha
                )
                for i in range(num_blocks):
                    self.v.blocks[i].attn = shared_lora
        else:
            # Each block gets its own
            for i in range(num_blocks):
                base_attn = self.v.blocks[i].attn
                if not isinstance(base_attn, LoRAQKVAttention):
                    lora_attn = LoRAQKVAttention(
                        timm_attn=base_attn,
                        dim=self.original_embedding_dim,
                        lora_target=self.lora_target,
                        lora_rank=self.lora_rank,
                        lora_alpha=self.lora_alpha
                    )
                    self.v.blocks[i].attn = lora_attn

    def freeze_base_model(self):
        """
        Freeze all giant ViT parameters. Then ONLY unfreeze:
          - The LoRA offsets (down_q, up_q, etc.)
          - The final classifier head
        """
        # Freeze everything
        for param in self.v.parameters():
            param.requires_grad = False

        # Now selectively unfreeze the LoRA offsets
        for i, block in enumerate(self.v.blocks):
            attn_module = block.attn
            if isinstance(attn_module, LoRAQKVAttention):
                # We only want 'down_q', 'up_q', 'down_v', 'up_v' to be trainable.
                # Timm's 'qkv' or 'proj' remain frozen.
                for name, p in attn_module.named_parameters():
                    # If it's a LoRA offset param
                    if name.startswith("down_") or name.startswith("up_"):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False

        # Also unfreeze final classifier
        for param in self.mlp_head.parameters():
            param.requires_grad = True

        print("Base model frozen. Only LoRA offsets + classifier are trainable.")

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1,1,input_fdim,input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16,16), stride=(fstride,tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def forward(self, x):
        B = x.shape[0]
        x = self.v.patch_embed(x)

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for i, blk in enumerate(self.v.blocks):
            residual = x
            x = blk.norm1(x)
            attn_out = blk.attn(x)  
            x = attn_out + residual
            x = blk.drop_path(x)

            residual = x
            x = blk.norm2(x)
            x = blk.mlp(x)
            x = x + residual
            x = blk.drop_path(x)

        x = (x[:, 0] + x[:, 1]) / 2
        x = self.mlp_head(x)
        return x
