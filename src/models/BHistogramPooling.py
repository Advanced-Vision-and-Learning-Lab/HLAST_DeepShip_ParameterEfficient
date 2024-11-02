import torch
import torch.nn as nn
import torch.nn.functional as F


class HistogramLayer(nn.Module):
    def __init__(self,in_channels,kernel_size,dim=1,num_bins=4,
                  stride=1,padding=0,normalize_count=True,normalize_bins = True,
                  count_include_pad=False, ceil_mode=False, df=3):

        # inherit nn.module
        super(HistogramLayer, self).__init__()
        
# class BoundedHistogramLayer(nn.Module):
#     def __init__(self, in_channels, num_bins=4):
#         super(BoundedHistogramLayer, self).__init__()
        self.in_channels = in_channels
        self.num_bins = num_bins

    def forward(self, x):
        batch_size, channels, seq_len = x.size()

        # Global average pooling to get global feature vector g
        g = x.mean(dim=2, keepdim=True)  # Shape: (batch_size, channels, 1)

        # Cosine similarity between g and each position in x
        cos_sim = F.cosine_similarity(x.unsqueeze(3), g.unsqueeze(2), dim=1)  # Shape: (batch_size, seq_len)

        # Reshape S to (batch_size * seq_len,)
        S = cos_sim.view(batch_size * seq_len)

        # Quantize S into N levels
        min_S = S.min()
        max_S = S.max()
        levels = torch.linspace(min_S, max_S, self.num_bins).to(x.device)  # Levels L

        # Quantization encoding
        E = torch.zeros(S.size(0), self.num_bins).to(x.device)
        
        for i in range(self.num_bins):
            L_n = levels[i]
            mask = (torch.abs(S - L_n) < 0.5 / self.num_bins).float()
            E[:, i] = (1 - torch.abs(S - L_n)) * mask

        # Reshape E back to (batch_size, seq_len, num_bins)
        E = E.view(batch_size, seq_len, self.num_bins)

        # Count the number of features belonging to each level
        count_map = E.sum(dim=1) / E.sum(dim=1).sum(dim=1, keepdim=True)  # Normalize counts

        # Use adaptive average pooling on count_map directly to get desired shape
        output = F.adaptive_avg_pool1d(count_map.unsqueeze(2), channels // self.num_bins).squeeze(2)  # Shape: [batch_size, num_bins]

        return output

# Example usage:
# layer = BoundedHistogramLayer(in_channels=768, num_bins=4)
# input_tensor = torch.randn(64, 768, 182)
# output = layer(input_tensor)
# print(output.shape)  # Expected output shape: [64, 4]