import torch
import torch.nn as nn
import torch.nn.functional as F

# Experimental Don't use
class TokenRegularization(torch.autograd.Function):
    @staticmethod
    def forward(ctx, key_tokens, value_tokens):
        ctx.save_for_backward(key_tokens, value_tokens)
        return key_tokens, value_tokens

    @staticmethod
    def backward(ctx, grad_k, grad_v):
        key_tokens, value_tokens = ctx.saved_tensors
        epsilon = 1e-5
        
        # Scale factors based on dimensionality
        dim_scale = 1.0 / math.sqrt(key_tokens.size(-1))
        lambda_var = 0.001 * dim_scale    # Variance regularization strength
        lambda_cov = 0.02 * dim_scale     # Increased covariance regularization
        lambda_div = 0.01 * dim_scale     # Diversity regularization strength
        
        def regularize_grad(tokens, grad):
            d = tokens.size(-1)  # embedding dimension
            n = tokens.size(0)   # number of tokens
            
            # Calculate covariance matrix
            tokens_centered = tokens - tokens.mean(0, keepdim=True)
            cov_matrix = torch.mm(tokens_centered.t(), tokens_centered) / (n - 1)
            
            # Variance regularization gradient
            diagonal = torch.rsqrt(cov_matrix.diagonal() + epsilon)
            diagonal = torch.clamp(diagonal, max=1.0)
            var_grad = diagonal * tokens_centered
            
            # Covariance regularization gradient (stronger push for orthogonality)
            cov_matrix.fill_diagonal_(0)  # Zero out diagonal
            cov_grad = torch.mm(tokens_centered, cov_matrix)
            
            # Diversity regularization (push tokens away from each other)
            similarity = torch.mm(tokens, tokens.t())
            similarity.fill_diagonal_(0)  # Don't push token away from itself
            # Normalize similarities to [-1, 1] range
            similarity = similarity / (torch.norm(tokens, dim=1, keepdim=True) @ torch.norm(tokens, dim=1, keepdim=True).t() + epsilon)
            # Generate repulsion forces based on similarity
            diversity_grad = torch.mm(similarity, tokens)
            
            # Combine gradients with proper scaling
            return (grad 
                   - lambda_var/(d*(n-1)) * var_grad  # Variance term
                   + lambda_cov/(d*(d-1)) * cov_grad  # Covariance term
                   - lambda_div/d * diversity_grad)    # Diversity term
        
        # Apply regularization to both key and value tokens
        grad_k = regularize_grad(key_tokens, grad_k)
        grad_v = regularize_grad(value_tokens, grad_v)
        
        return grad_k, grad_v

class TokenLinear(nn.Module):
    def __init__(self, in_features, out_features, num_tokens):
        super().__init__()
        self.num_tokens = num_tokens
        self.in_features = in_features 
        self.out_features = out_features
        
        # Core token parameters
        self.key_tokens = nn.Parameter(torch.empty(num_tokens, in_features))
        self.value_tokens = nn.Parameter(torch.empty(num_tokens, out_features))
                        
        # Better initialization
        std = 1.0 / math.sqrt(in_features)
        nn.init.normal_(self.key_tokens, mean=0.0, std=std)
        nn.init.normal_(self.value_tokens, mean=0.0, std=std)

    def mod_softmax(self, attn_map):
        out = (attn_map * (attn_map.size(-1) ** 0.5)) / torch.sqrt(attn_map.square().sum(-1, keepdim=True))
        return torch.nn.functional.gelu(out)

    def forward(self, x):
        # Apply regularization through custom autograd function
        key_tokens, value_tokens = TokenRegularization.apply(self.key_tokens, self.value_tokens)
        
        # Token-based transformation
        similarity = torch.matmul(x, key_tokens.T)
        weights = self.mod_softmax(similarity)
        token_out = torch.matmul(weights, value_tokens)
        
        return token_out
