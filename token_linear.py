import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenLinear(nn.Module):
    def __init__(self, in_features, out_features, num_tokens, rank=32):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_features = in_features 
        self.out_features = out_features
        self.rank = rank
        
        # Core token parameters
        self.key_tokens = nn.Parameter(torch.empty(num_tokens, in_features))
        self.value_tokens = nn.Parameter(torch.empty(num_tokens, out_features))
        
        # Low rank projection matrices for transforming existing tokens
        self.key_down = nn.Parameter(torch.empty(in_features, rank))
        self.key_up = nn.Parameter(torch.empty(rank, in_features))
        self.value_down = nn.Parameter(torch.empty(out_features, rank))
        self.value_up = nn.Parameter(torch.empty(rank, out_features))
                
        # Better initialization
        std = 1.0 / math.sqrt(in_features)
        nn.init.normal_(self.key_tokens, mean=0.0, std=std)
        nn.init.normal_(self.value_tokens, mean=0.0, std=std)
        
        # Initialize low rank projections
        for param in [self.key_down, self.key_up, self.value_down, self.value_up]:
            nn.init.normal_(param, mean=0.0, std=std)

    def mod_softmax(self, attn_map):
        out = (attn_map * (attn_map.size(-1) ** 0.5)) / torch.sqrt(attn_map.square().sum(-1, keepdim=True))
        return torch.nn.functional.gelu(out)

    def forward(self, x):
        # Create transformed versions of tokens through low-rank projections
        key_transformed = F.gelu(torch.matmul(torch.matmul(self.key_tokens, self.key_down), self.key_up))
        value_transformed = F.gelu(torch.matmul(torch.matmul(self.value_tokens, self.value_down), self.value_up))
        
        # Combine original and transformed tokens
        combined_keys = torch.cat([self.key_tokens, key_transformed], dim=0)
        combined_values = torch.cat([self.value_tokens, value_transformed], dim=0)
        
        # Token-based transformation
        similarity = torch.matmul(x, combined_keys.T)
        weights = self.mod_softmax(similarity)
        token_out = torch.matmul(weights, combined_values)
        
        return token_out
