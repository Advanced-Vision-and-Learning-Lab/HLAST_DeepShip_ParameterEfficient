import torch.nn as nn
from transformers import ASTModel
import torch
class ASTLinearProbe(nn.Module):
    def __init__(self, num_labels, max_length, num_mel_bins, model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        self.model = ASTModel.from_pretrained(model_ckpt, max_length=max_length, num_mel_bins=num_mel_bins, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        
        # Freeze all parameters of the base model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Simple linear layer for classification
        self.classifier = nn.Linear(self.model_config.hidden_size, num_labels)

    def forward(self, input_values):
        with torch.no_grad():
            hidden_states = self.model(input_values)[0]
        
        # Use the [CLS] token representation (first token)
        pooled_output = hidden_states[:, 0]
        logits = self.classifier(pooled_output)
        return logits