import torch
import torch.nn as nn
from transformers import ASTModel

class AdapterLayer(nn.Module):
    def __init__(self, input_size, reduction_rate):
        super().__init__()
        self.adapter_size = input_size // reduction_rate
        self.down = nn.Linear(input_size, self.adapter_size)
        self.up = nn.Linear(self.adapter_size, input_size)
        self.act = nn.ReLU()
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.up(self.act(self.down(x)))

class ASTAdapter(nn.Module):
    def __init__(self, num_labels, max_length, num_mel_bins, adapter_size, 
                 adapter_location='mhsa_ffn', adapter_shared=True, adapter_mode='parallel', 
                 model_ckpt="MIT/ast-finetuned-audioset-10-10-0.4593"):
        super().__init__()
        
        self.model = ASTModel.from_pretrained(model_ckpt, max_length=max_length, num_mel_bins=num_mel_bins, ignore_mismatched_sizes=True)
        self.model_config = self.model.config
        self.adapter_location = adapter_location
        self.adapter_mode = adapter_mode
        self.adapter_shared = adapter_shared
        self.reduction_rate = adapter_size
        
        self.adapters = nn.ModuleDict()
        num_layers = len(self.model.encoder.layer)
        
        print(f"Initializing ASTAdapter with the following configuration:")
        print(f"Adapter location: {adapter_location}")
        print(f"Adapter mode: {adapter_mode}")
        print(f"Adapter shared: {adapter_shared}")
        print(f"Reduction Rate: {self.reduction_rate}")
        print(f"Number of transformer layers: {num_layers}")
        
        if adapter_shared:
            if 'mhsa' in adapter_location:
                self.adapters['mhsa'] = AdapterLayer(self.model_config.hidden_size, self.reduction_rate)
                print(f"Added shared MHSA adapter with reduction rate {self.reduction_rate}")
            if 'ffn' in adapter_location:
                self.adapters['ffn'] = AdapterLayer(self.model_config.hidden_size, self.reduction_rate)
                print(f"Added shared FFN adapter with reduction rate {self.reduction_rate}")
        else:
            if 'mhsa' in adapter_location:
                self.adapters['mhsa'] = nn.ModuleList([AdapterLayer(self.model_config.hidden_size, self.reduction_rate) for _ in range(num_layers)])
                print(f"Added {num_layers} non-shared MHSA adapters with reduction rate {self.reduction_rate}")
            if 'ffn' in adapter_location:
                self.adapters['ffn'] = nn.ModuleList([AdapterLayer(self.model_config.hidden_size, self.reduction_rate) for _ in range(num_layers)])
                print(f"Added {num_layers} non-shared FFN adapters with reduction rate {self.reduction_rate}")
        
        self.classifier = nn.Linear(self.model_config.hidden_size, num_labels)
        self.freeze_base_model()

    def freeze_base_model(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for adapter in self.adapters.values():
            for param in adapter.parameters():
                param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("Base model frozen. Only adapters and classifier are trainable.")
    
    def forward(self, input_values):
        hidden_states = self.model.embeddings(input_values)
        
        for i, layer in enumerate(self.model.encoder.layer):
            residual = hidden_states
            hidden_states = layer.layernorm_before(hidden_states)
            
            # MHSA sublayer
            attn_output = layer.attention(hidden_states)[0]
            if 'mhsa' in self.adapter_location:
                adapter = self.adapters['mhsa'] if self.adapter_shared else self.adapters['mhsa'][i]
                if self.adapter_mode == 'parallel':
                    attn_output = attn_output + adapter(residual)
                else:  # sequential
                    attn_output = adapter(attn_output)
            hidden_states = residual + attn_output
            
            # FFN sublayer
            residual = hidden_states
            hidden_states = layer.layernorm_after(hidden_states)
            ffn_output = layer.intermediate(hidden_states)
            ffn_output = layer.output(ffn_output, hidden_states)  # Pass both ffn_output and hidden_states
            
            if 'ffn' in self.adapter_location:
                adapter = self.adapters['ffn'] if self.adapter_shared else self.adapters['ffn'][i]
                if self.adapter_mode == 'parallel':
                    ffn_output = ffn_output + adapter(residual)
                else:  # sequential
                    ffn_output = adapter(ffn_output)
            
            hidden_states = residual + ffn_output
    
        hidden_states = self.model.layernorm(hidden_states)
        logits = self.classifier(hidden_states[:, 0])
        return logits