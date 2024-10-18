import torch
import torch.nn as nn

class ProgressiveTokenizationModule(nn.Module):
    def __init__(self, input_channels=1):
        super(ProgressiveTokenizationModule, self).__init__()

        # T-F Split layers
        self.tf_split1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=4, padding=2)
        self.tf_split2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.tf_split3 = nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1)

        # Token attention calculation (MHSA with H=1)
        self.attention1 = nn.MultiheadAttention(64, num_heads=1)
        self.attention2 = nn.MultiheadAttention(128, num_heads=1)
        self.attention3 = nn.MultiheadAttention(192, num_heads=1)


    def forward(self, x):
        # First T-F Split and Attention
        x = self.tf_split1(x)
        b, c, h, w = x.shape
        x_flat = x.flatten(2).permute(2, 0, 1)
        x_att, _ = self.attention1(x_flat, x_flat, x_flat)
        x = x_att.permute(1, 2, 0).view(b, c, h, w)

        # Second T-F Split and Attention
        x = self.tf_split2(x)
        b, c, h, w = x.shape
        x_flat = x.flatten(2).permute(2, 0, 1)
        x_att, _ = self.attention2(x_flat, x_flat, x_flat)
        x = x_att.permute(1, 2, 0).view(b, c, h, w)

        # Third T-F Split and Attention
        x = self.tf_split3(x)
        b, c, h, w = x.shape
        x_flat = x.flatten(2).permute(2, 0, 1)
        x_att, _ = self.attention3(x_flat, x_flat, x_flat)
        x = x_att.permute(1, 2, 0).view(b, c, h, w)

        

        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, 1, 192, h * w)

        return x