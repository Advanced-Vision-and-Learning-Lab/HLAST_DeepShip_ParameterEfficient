import torch
import torch.nn as nn
import os
import wget
os.environ['TORCH_HOME'] = 'pretrained_models'
import timm
from timm.models.layers import to_2tuple, trunc_normal_

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

class ASTBias(nn.Module):
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, 
                 imagenet_pretrain=True, audioset_pretrain=True, model_size='base384', verbose=True,
                 bias_mode='full'):
        """
        bias_mode options:
            'full'  : unfreeze all bias parameters in the base model (default, as before)
            'query' : unfreeze only the query bias in attention (for combined qkv layers, a hook zeros key/value grads)
            'mlp'   : unfreeze only the bias parameters in the second (output) linear layer of the MLP.
        """
        super(ASTBias, self).__init__()
        self.bias_mode = bias_mode  # new argument to select which bias parameters to tune

        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5, the code might not be compatible with newer versions.'

        if verbose:
            print('---------------AST Model Summary---------------')
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain), str(audioset_pretrain)))
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

            # automatically get the intermediate shape
            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            # the linear projection layer
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            # the positional embedding
            if imagenet_pretrain:
                # get the positional embedding from deit model, skip the first two tokens (cls token and distillation token),
                # reshape it to original 2D shape (e.g., 24x24).
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim)
                new_pos_embed = new_pos_embed.transpose(1, 2).reshape(1, self.original_embedding_dim, self.oringal_hw, self.oringal_hw)
                # cut (from middle) or interpolate the second dimension of the positional embedding
                if t_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, :, int(self.oringal_hw/2) - int(t_dim/2): int(self.oringal_hw/2) - int(t_dim/2) + t_dim]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(self.oringal_hw, t_dim), mode='bilinear')
                # cut (from middle) or interpolate the first dimension of the positional embedding
                if f_dim <= self.oringal_hw:
                    new_pos_embed = new_pos_embed[:, :, int(self.oringal_hw/2) - int(f_dim/2): int(self.oringal_hw/2) - int(f_dim/2) + f_dim, :]
                else:
                    new_pos_embed = torch.nn.functional.interpolate(new_pos_embed, size=(f_dim, t_dim), mode='bilinear')
                # flatten the positional embedding
                new_pos_embed = new_pos_embed.reshape(1, self.original_embedding_dim, num_patches).transpose(1,2)
                # concatenate the above positional embedding with the cls token and distillation token of the deit model.
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed], dim=1))
            else:
                # if not use imagenet pretrained model, just randomly initialize a learnable positional embedding
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        # now load a model that is pretrained on both ImageNet and AudioSet
        elif audioset_pretrain == True:
            if audioset_pretrain and not imagenet_pretrain:
                raise ValueError('Currently model pretrained on only AudioSet is not supported; please set imagenet_pretrain = True.')
            if model_size != 'base384':
                raise ValueError('Currently only has base384 AudioSet pretrained model.')
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if not os.path.exists('pretrained_models/audioset_10_10_0.4593.pth'):
                # this model performs 0.4593 mAP on the AudioSet eval set
                audioset_mdl_url = 'https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1'
                wget.download(audioset_mdl_url, out='pretrained_models/audioset_10_10_0.4593.pth')
            sd = torch.load('pretrained_models/audioset_10_10_0.4593.pth', map_location=device, weights_only=True)
            
            # Here, we just initialize the same ASTBias class (which won't have adapters) to load weights.
            audio_model = ASTBias(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024,
                                   imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            
            print(f"\nNumber of transformer blocks: {len(self.v.blocks)}")

            # the classification head
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(self.original_embedding_dim),
                nn.Linear(self.original_embedding_dim, label_dim)
            )

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            num_patches = f_dim * t_dim
            self.v.patch_embed.num_patches = num_patches
            if verbose:
                print('frequncey stride={:d}, time stride={:d}'.format(fstride, tstride))
                print('number of patches={:d}'.format(num_patches))

            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            # if the input sequence length is larger than the original AudioSet (10s), then cut the positional embedding
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
            return

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.original_embedding_dim),
            nn.Linear(self.original_embedding_dim, label_dim)
        )
            
        self.freeze_base_model()

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        f_dim = test_out.shape[2]
        t_dim = test_out.shape[3]
        return f_dim, t_dim

    def freeze_base_model(self):
        # Freeze all parameters of the base model first.
        for param in self.v.parameters():
            param.requires_grad = False

        if self.bias_mode == 'full':
            # Unfreeze every bias parameter (as in the original implementation).
            for name, param in self.v.named_parameters():
                if ".bias" in name:
                    param.requires_grad = True
            mode_msg = "all bias parameters"
        elif self.bias_mode == 'query':
            # Unfreeze only the query bias inside the attention modules.
            # In many implementations, the attention module uses a combined qkv layer.
            # Here we unfreeze that qkv bias and register a hook to zero out the gradient for key and value parts.
            def make_query_hook(dim):
                def hook(grad):
                    grad_clone = grad.clone()
                    grad_clone[dim: 2*dim] = 0  # zero out key
                    grad_clone[2*dim: 3*dim] = 0  # zero out value
                    return grad_clone
                return hook

            for name, param in self.v.named_parameters():
                if "attn.qkv.bias" in name:
                    param.requires_grad = True
                    dim = param.shape[0] // 3
                    param.register_hook(make_query_hook(dim))
            mode_msg = "query bias parameters (only the query part will be updated)"
        elif self.bias_mode == 'mlp':
            # Unfreeze only the bias of the second (output) linear layer in the MLP module.
            for name, param in self.v.named_parameters():
                if "mlp.fc2.bias" in name:
                    param.requires_grad = True
            mode_msg = "second MLP bias parameters"
        else:
            raise ValueError("Invalid bias_mode. Choose from 'full', 'query', or 'mlp'.")

        # Finally, unfreeze all parameters in the final classification head.
        for param in self.mlp_head.parameters():
            param.requires_grad = True

        print(f"Base model frozen except for {mode_msg}. Only selected biases and classifier are trainable.")

    def forward(self, x):
        B = x.shape[0]

        x = self.v.patch_embed(x)

        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.v.pos_embed
        x = self.v.pos_drop(x)

        for i, blk in enumerate(self.v.blocks):
            # Multi-head self-attention (MHSA) sublayer.
            residual = x
            x = blk.norm1(x)
            attn_out = blk.attn(x)
            x = attn_out + residual
            x = blk.drop_path(x)

            # Feed-forward (FFN) sublayer.
            residual = x
            x = blk.norm2(x)
            ffn_out = blk.mlp(x)
            x = ffn_out + residual
            x = blk.drop_path(x)

        x = (x[:, 0] + x[:, 1]) / 2
        x = self.mlp_head(x)

        return x
