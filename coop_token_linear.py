import torch
import torch.nn as nn
import torch.nn.functional as F


class CoopCompeteTokenLinear(nn.Module):
    def __init__(self, in_features, out_features, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_features = in_features 
        self.out_features = out_features
        
        # Core token parameters
        self.key_tokens = nn.Parameter(torch.empty(num_tokens, in_features))
        self.value_tokens = nn.Parameter(torch.empty(num_tokens, out_features))
        
        # Token interaction parameters
        self.competition = nn.Parameter(torch.empty(num_tokens, 1))  # How much each token competes
        self.cooperation = nn.Parameter(torch.empty(num_tokens, 1))  # How much each token cooperates
        
        std = 1.0 / math.sqrt(in_features)
        nn.init.normal_(self.key_tokens, mean=0.0, std=std)
        nn.init.normal_(self.value_tokens, mean=0.0, std=std)
        nn.init.uniform_(self.competition, 0.0, 1.0)
        nn.init.uniform_(self.cooperation, 0.0, 1.0)

    def mod_softmax(self, attn_map):
        out = (attn_map * (attn_map.size(-1) ** 0.5)) / torch.sqrt(attn_map.square().sum(-1, keepdim=True))
        return torch.nn.functional.gelu(out)

    def forward(self, x):  # x is [batch, seq_len, embed]
        similarity = torch.matmul(x, self.key_tokens.T)  # [batch, seq_len, num_tokens]
        
        # Token interaction dynamics based on key similarity
        token_interactions = torch.matmul(self.key_tokens, self.key_tokens.T)  # [num_tokens, num_tokens]
        token_interactions = self.mod_softmax(token_interactions)  # normalize interactions
        
        # Competition: inhibit tokens, modulated by token similarity
        competition_mask = (self.competition * self.competition.T).sigmoid() * token_interactions
        inhibition = torch.matmul(similarity, competition_mask)  # [batch, seq_len, num_tokens]
        
        # Cooperation: boost tokens, modulated by token dissimilarity
        cooperation_mask = (self.cooperation * self.cooperation.T).sigmoid() * (1 - token_interactions)
        boost = torch.matmul(similarity, cooperation_mask)  # [batch, seq_len, num_tokens]
        
        evolved_similarity = similarity - inhibition + boost
        
        # Final attention and output
        weights = self.mod_softmax(evolved_similarity)  # [batch, seq_len, num_tokens]
        token_out = torch.matmul(weights, self.value_tokens)  # [batch, seq_len, out_features]
        
        return token_out
