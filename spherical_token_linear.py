import torch
import torch.nn as nn
import torch.nn.functional as F

class SphericalTokenLinear(nn.Module):
    def __init__(self, in_features, out_features, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_features = in_features 
        self.out_features = out_features
        
        # Core token parameters
        self.key_tokens = nn.Parameter(torch.empty(num_tokens, in_features))
        self.value_tokens = nn.Parameter(torch.empty(num_tokens, out_features))
        
        # Learnable temperature per token
        self.temp = nn.Parameter(torch.empty(num_tokens, 1))
        
        std = 1.0 / math.sqrt(in_features)
        nn.init.normal_(self.key_tokens, mean=0.0, std=std)
        nn.init.normal_(self.value_tokens, mean=0.0, std=std)
        nn.init.constant_(self.temp, 1.0)

    def mod_softmax(self, attn_map):
        out = (attn_map * (attn_map.size(-1) ** 0.5)) / torch.sqrt(attn_map.square().sum(-1, keepdim=True))
        return torch.nn.functional.gelu(out)

    def forward(self, x):  # x is [batch, seq_len, embed]
        # Normalize inputs and keys to unit sphere along embedding dimension
        x_norm = F.normalize(x, p=2, dim=-1)  # [batch, seq_len, embed]
        k_norm = F.normalize(self.key_tokens, p=2, dim=-1)  # [num_tokens, embed]
        
        # Compute cosine similarity with learned temperature
        similarity = torch.matmul(x_norm, k_norm.T)  # [batch, seq_len, num_tokens]
        similarity = similarity * self.temp.sigmoid().view(1, 1, -1)
        
        # Apply modified softmax with spherical constraints
        weights = self.mod_softmax(similarity)  # [batch, seq_len, num_tokens]
        token_out = torch.matmul(weights, self.value_tokens)  # [batch, seq_len, out_features]
        
        return token_out
