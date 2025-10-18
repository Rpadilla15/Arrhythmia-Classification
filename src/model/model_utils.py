import torch
import torch.nn as nn
import math


# -----------------
# Positional Embedding for 1D signal
# -----------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [B, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# -----------------
# Learned Positional Embedding (Decoder)
# -----------------
class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings without dropout for decoder."""
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x):
        # x: [B, seq_len, d_model]
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]
    
# -----------------
# Patch Embedding for 1D signal
# -----------------
class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels=1, emb_size=64, patch_size=10):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_emb_generator = PositionalEncoding(emb_size)

    def forward(self, x):
        # x: [B, C, T]
        x = self.proj(x)             # [B, emb_size, T//patch_size]
        x = x.permute(0, 2, 1)       # [B, num_patches, emb_size]

        # Generate sinusoidal pos embedding dynamically
        x = self.pos_emb_generator(x)

        # Add CLS token
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, emb_size]
        x = torch.cat([cls_token, x], dim=1)  # [B, num_patches+1, emb_size]

        return x

def patch_proj(x, patch_size):
        """
        Transforms a tensor from [B, C, T] to [B, T//patch_size, C * patch_size] 
        using a single chained operation.
        """
        B, C, T = x.shape
        num_patches = T // patch_size

        output_tensor = x.view(B, C, num_patches, patch_size) \
                                .permute(0, 2, 1, 3) \
                                .reshape(B, num_patches, C * patch_size)
        return output_tensor

def patch_proj_inv(tensor_transformed, C, patch_size):
    """
    Transforms a tensor from [B, N, F] back to [B, C, T].
    where N = T//length_chunk and F = C * length_chunk.

    Args:
        tensor_transformed (torch.Tensor): The input tensor.
        C (int): The original number of channels.
        length_chunk (int): The original length of the time chunks.
    """
    B = tensor_transformed.shape[0]
    num_patches = tensor_transformed.shape[1]
    
    # 1. view F (C*L) into (C, L)
    # 2. permute to move C back to the 2nd dimension
    # 3. reshape the two temporal dimensions (num_patches and L) into T
    output_tensor_bct = tensor_transformed.view(B, num_patches, C, patch_size) \
                                          .permute(0, 2, 1, 3) \
                                          .reshape(B, C, -1) # -1 infers the final T dimension
    return output_tensor_bct

