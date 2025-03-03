import torch
import torch.nn as nn
import os
import wget
import math
os.environ['TORCH_HOME'] = 'pretrained_models'
import timm
from timm.models.layers import to_2tuple, trunc_normal_

# ------------------------------------------------------------
# LoRA_qkv Module with Optional "q" or "qv" Update Modes
# ------------------------------------------------------------
class LoRA_qkv(nn.Module):
    def __init__(self, linear_layer: nn.Linear, r: int = 4, alpha: float = 1.0, update_mode: str = "qv"):
        """
        Wraps the original qkv linear layer (which outputs concatenated Q, K, V) and adds low-rank updates.
        
        Args:
            linear_layer: The original qkv projection layer (output shape: [B, N, 3*dim]).
            r: The LoRA rank.
            alpha: Scaling factor for the LoRA update.
            update_mode: Either "qv" (update both query and value) or "q" (update query only).
        """
        super(LoRA_qkv, self).__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        assert self.out_features % 3 == 0, "Expected out_features to be divisible by 3 (query, key, value)"
        self.dim = self.out_features // 3  # dimension for each of Q, K, V
        self.r = r
        self.alpha = alpha
        self.update_mode = update_mode.lower()
        if self.update_mode not in ["q", "qv"]:
            raise ValueError("update_mode must be either 'q' or 'qv'")
        self.scaling = alpha / r if r > 0 else 0

        # Freeze and store the original weight and bias.
        self.register_buffer("weight", linear_layer.weight.detach().clone())
        if linear_layer.bias is not None:
            self.register_buffer("bias", linear_layer.bias.detach().clone())
        else:
            self.bias = None

        if r > 0:
            # Create LoRA parameters for the query part (always)
            self.lora_A_q = nn.Parameter(torch.zeros(self.dim, r))
            self.lora_B_q = nn.Parameter(torch.zeros(r, self.in_features))
            nn.init.kaiming_uniform_(self.lora_B_q, a=math.sqrt(5))
            nn.init.zeros_(self.lora_A_q)
            if self.update_mode == "qv":
                # Also create LoRA parameters for the value part.
                self.lora_A_v = nn.Parameter(torch.zeros(self.dim, r))
                self.lora_B_v = nn.Parameter(torch.zeros(r, self.in_features))
                nn.init.kaiming_uniform_(self.lora_B_v, a=math.sqrt(5))
                nn.init.zeros_(self.lora_A_v)
            else:
                self.lora_A_v = None
                self.lora_B_v = None
        else:
            self.lora_A_q = None
            self.lora_B_q = None
            self.lora_A_v = None
            self.lora_B_v = None

    def forward(self, x):
        # Compute the original qkv output (frozen weights)
        original_out = torch.nn.functional.linear(x, self.weight, self.bias)  # shape: [B, N, 3*dim]
        if self.r > 0:
            # Compute low-rank update for the query part.
            delta_q = (x @ self.lora_B_q.T) @ self.lora_A_q.T  # shape: [B, N, dim]
            delta_q = delta_q * self.scaling
            out = original_out.clone()
            out[..., :self.dim] += delta_q  # add update to query part
            
            # Compute and add update for value part if update_mode=="qv"
            if self.update_mode == "qv":
                delta_v = (x @ self.lora_B_v.T) @ self.lora_A_v.T  # shape: [B, N, dim]
                delta_v = delta_v * self.scaling
                out[..., 2*self.dim:3*self.dim] += delta_v  # add update to value part
            return out
        else:
            return original_out

# ------------------------------------------------------------
# PatchEmbed: override timm's input shape restriction.
# ------------------------------------------------------------
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


class ASTLoRA(nn.Module):
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 imagenet_pretrain=True, audioset_pretrain=True, model_size='base384', verbose=True, 
                 lora_shared=True, lora_rank=4, lora_update_mode="qv"):
        """
        Args:
            label_dim: Number of output classes.
            fstride, tstride: Strides for the projection.
            input_fdim, input_tdim: Input feature dimensions.
            imagenet_pretrain, audioset_pretrain: Flags to control model loading.
            model_size: one of 'tiny224', 'small224', 'base224', 'base384'.
            verbose: Whether to print status messages.
            lora_shared: If True, all transformer blocks share the same LoRA_qkv module.
            lora_rank: LoRA rank (r in the paper).
            lora_update_mode: Either "qv" (update query and value) or "q" (update query only).
        """
        super(ASTLoRA, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain), str(audioset_pretrain)))
        # Override timm input shape restriction.
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

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

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print('frequency stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain:
                new_proj.weight = nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if imagenet_pretrain:
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach() \
                    .reshape(1, self.original_num_patches, self.original_embedding_dim) \
                    .transpose(1, 2) \
                    .reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw / 2) - int(t_dim / 2): int(self.oringal_hw / 2) - int(t_dim / 2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw / 2) - int(f_dim / 2): int(self.oringal_hw / 2) - int(f_dim / 2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # ------------------------------
        # AudioSet pretraining.
        # attention module by replacing its qkv layer with a LoRA_qkv that uses the chosen update mode.
        # ------------------------------
        elif audioset_pretrain == True:
            if audioset_pretrain == True and imagenet_pretrain == False:
                raise ValueError('Currently a model pretrained only on AudioSet is not supported, please set imagenet_pretrain=True.')
            if model_size != 'base384':
                raise ValueError('Currently only the base384 AudioSet pretrained model is supported.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not os.path.exists('pretrained_models/audioset_10_10_0.4593.pth'):
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('pretrained_models/audioset_10_10_0.4593.pth', map_location=device, weights_only=True)
            audio_model = ASTLoRA(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                                     imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            
            print(f"\nNumber of transformer blocks: {len(self.v.blocks)}")
            
            # Patch each transformer's attention module:
            # Replace qkv with LoRA_qkv (either shared or unique per block).
            if lora_shared:
                shared_lora_qkv = None
                for blk in self.v.blocks:
                    if shared_lora_qkv is None:
                        shared_lora_qkv = LoRA_qkv(blk.attn.qkv, r=lora_rank, alpha=1.0, update_mode=lora_update_mode)
                    blk.attn.qkv = shared_lora_qkv
            else:
                for blk in self.v.blocks:
                    blk.attn.qkv = LoRA_qkv(blk.attn.qkv, r=lora_rank, alpha=1.0, update_mode=lora_update_mode)
   
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
    
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach() \
                .reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
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
        
        # Unfreeze only the LoRA parameters inside qkv.
        for blk in self.v.blocks:
            for name, param in blk.attn.qkv.named_parameters():
                if 'lora' in name:
                    param.requires_grad = True
        
        for param in self.mlp_head.parameters():
            param.requires_grad = True
        
        print("Base model frozen. Only LoRA parameters and classifier are trainable.")
    
    def forward(self, x):
        B = x.shape[0]

        x = self.v.patch_embed(x)

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for blk in self.v.blocks:
            residual = x
            x = blk.norm1(x)
            x = blk.attn(x)   
            x = residual + x
            x = blk.drop_path(x)

            residual = x
            x = blk.norm2(x)
            x = blk.mlp(x)
            x = residual + x
            x = blk.drop_path(x)

        x = (x[:, 0] + x[:, 1]) / 2
        x = self.mlp_head(x)

        return x 
