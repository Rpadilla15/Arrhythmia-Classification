import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Add CLS token
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, emb_size]
        x = torch.cat([cls_token, x], dim=1)  # [B, num_patches+1, emb_size]

        # Generate sinusoidal pos embedding dynamically
        x = self.pos_emb_generator(x)

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

# -----------------
# Transformer Autoencoder (REVISED for Masking)
# -----------------
class TransformerAutoencoder(nn.Module):
    def __init__(self, in_channels=2, emb_size=64, patch_size=10, num_layers=2, nhead=4, max_len=5000):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_size = emb_size
        max_patches = max_len // patch_size
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.emb_size))

        # Encoder components
        self.encoder_embed = PatchEmbedding1D(self.in_channels, self.emb_size, patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=nhead, dim_feedforward=4*self.emb_size, batch_first=True
            )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers*2)

        # Decoder components
        self.decoder_pos_embed = LearnedPositionalEmbedding(max_patches, self.emb_size)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=nhead, dim_feedforward=4*self.emb_size, batch_first=True
            )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Reconstruct layer: Projects token dim (emb_size) back to patch dim (C * P)
        self.reconstruct = nn.Linear(self.emb_size, self.in_channels * patch_size)

    def normalize_patches(self, patches, Stats = True):
        mean = patches.mean(dim=-1, keepdim=True)
        std = patches.std(dim=-1, keepdim=True)
        if Stats:
            return (patches - mean) / (std + 1e-8), mean, std
        return (patches - mean) / (std + 1e-8)

    @torch.no_grad()
    def encode(self, x):
        """Return encoder representation (no masking, no decoder)."""
        self.eval()
        patches = self.encoder_embed(x)
        memory = self.encoder(patches)
        return memory[:, 0, :]  # CLS token

    def forward(self, x, mask_ratio=0.75):
        """
        Args:
            x: [B, C, T] input 1D signal
            mask_ratio: fraction of patches to mask

        Returns:
            dict with recon, loss, embedding (CLS), mask
        """
        B, C, T = x.shape
        device = x.device
        num_patches = T // self.patch_size

        # ------------------------------------------------------
        # 1. Get patch embeddings (with CLS + pos emb)
        # ------------------------------------------------------
        patches = self.encoder_embed(x)  # [B, num_patches+1, emb]
        cls_token, patch_tokens = patches[:, :1, :], patches[:, 1:, :]  # split CLS vs others

        # ------------------------------------------------------
        # 2. Generate random mask indices per sample
        # ------------------------------------------------------
        noise = torch.rand(B, num_patches, device=device)               # random noise for shuffling
        ids_shuffle = torch.argsort(noise, dim=1)                       # ascend per sample
        num_mask = int(num_patches * mask_ratio)
        ids_keep = ids_shuffle[:, num_mask:]                            # kept patch indices
        ids_mask = ids_shuffle[:, :num_mask]                            # masked patch indices

        # ------------------------------------------------------
        # 3. Encoder: feed only visible tokens (+ CLS)
        # ------------------------------------------------------
        # gather visible patches in original order (sorted)
        ids_keep_sorted, _ = torch.sort(ids_keep, dim=1)
        visible_tokens = torch.gather(
            patch_tokens,
            dim=1,
            index=ids_keep_sorted.unsqueeze(-1).expand(-1, -1, self.emb_size)
        )  # [B, num_visible, emb]

        # concat CLS
        z_masked = torch.cat([cls_token, visible_tokens], dim=1)  # [B, 1+num_visible, emb]
        memory = self.encoder(z_masked)                           # [B, 1+num_visible, emb]

        # ------------------------------------------------------
        # 4. Prepare decoder input (full sequence length)
        # ------------------------------------------------------
        # start with mask tokens for all patches
        decoder_input = self.mask_token.repeat(B, num_patches, 1)  # [B, num_patches, emb]

        # insert encoded visible tokens into their original positions
        encoded_visible = memory[:, 1:, :]  # drop CLS
        decoder_input.scatter_(
            1,
            ids_keep_sorted.unsqueeze(-1).expand(-1, -1, self.emb_size),
            encoded_visible
        )

        # Add learned positional embeddings
        decoder_input = self.decoder_pos_embed(decoder_input)

        # ------------------------------------------------------
        # 5. Decode (reconstruct)
        # ------------------------------------------------------
        decoded = self.decoder(decoder_input)  # [B, num_patches, emb]
        predicted_patches = self.reconstruct(decoded)  # [B, num_patches, C*P]

        # ------------------------------------------------------
        # 6. Compute reconstruction loss only on masked patches
        # ------------------------------------------------------
        true_patches = patch_proj(x, self.patch_size)  # [B, num_patches, C*P]
        ids_mask_sorted, _ = torch.sort(ids_mask, dim=1)
        masked_pred = torch.gather(
            predicted_patches,
            dim=1,
            index=ids_mask_sorted.unsqueeze(-1).expand(-1, -1, C * self.patch_size)
        )
        masked_true = torch.gather(
            true_patches,
            dim=1,
            index=ids_mask_sorted.unsqueeze(-1).expand(-1, -1, C * self.patch_size)
        )

        loss = F.mse_loss(masked_pred, masked_true)

        # ------------------------------------------------------
        # 7. Return
        # ------------------------------------------------------
        return {
            'recon': predicted_patches,
            'loss': loss,
            'embedding': memory[:, 0, :],  # CLS token
            'mask': (ids_mask, ids_keep),
        }


if __name__ == "__main__":
    x = torch.randn(2, 1, 100)
    model = TransformerAutoencoder()
    out = model(x)
    print(out['loss'], out['recon'].shape, out['embedding'].shape)
