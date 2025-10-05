import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------
# Positional Embedding for 1D signal
# -----------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, n_patches, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pos = torch.arange(n_patches, dtype=torch.float, device=device)  # [n_patches]
        i = torch.arange(self.dim // 2, dtype=torch.float, device=device)  # [dim/2]

        angles = pos[:, None] / (10000 ** (2 * i / self.dim))  # [n_patches, dim/2]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # [n_patches, dim]
        return emb[None, :, :]  # [1, n_patches, dim]


# -----------------
# Patch Embedding for 1D signal
# -----------------
class PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels=1, emb_size=64, patch_size=10):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_emb_generator = SinusoidalPosEmb(emb_size)

    def forward(self, x):
        # x: [B, C, T]
        x = self.proj(x)             # [B, emb_size, T//patch_size]
        x = x.permute(0, 2, 1)       # [B, num_patches, emb_size]
        
        # Add CLS token
        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # [B, 1, emb_size]
        x = torch.cat([cls_token, x], dim=1)  # [B, num_patches+1, emb_size]

        # Generate sinusoidal pos embedding dynamically
        pos_emb = self.pos_emb_generator(x.size(1), device=x.device)  # [1, num_patches+1, emb_size]

        return x + pos_emb


# -----------------
# Masking Utility
# -----------------

def generate_mask(batch_size: int, num_patches: int, mask_ratio: float, device="cpu"):
    """
    Generate a mask tensor for a batch of patch embeddings.
    
    Args:
        batch_size (int): Number of samples in the batch.
        num_patches (int): Number of patches per sample (excluding extra one).
        mask_ratio (float): Fraction of patches to mask (0.0 - 1.0).
        device (torch.device or str, optional): Device for the tensor.
    
    Returns:
        torch.Tensor: Mask tensor of shape [batch_size, num_patches + 1],
                      where mask[i, j] = 1 means "masked", and 0 means "keep".
                      The first column (extra patch) is always 0 (never masked).
    """

    # Compute number of patches to mask per sample (same for all)
    num_mask = int(num_patches * mask_ratio)

    # Create random noise to shuffle patches per batch
    noise = torch.rand(batch_size, num_patches, device=device)

    # Sort to get mask positions per sample
    ids_shuffle = torch.argsort(noise, dim=1)
    mask = torch.zeros(batch_size, num_patches, device=device)
    mask.scatter_(1, ids_shuffle[:, :num_mask], 1)  # Set masked patches to 1

    # Add extra patch at beginning (never masked)
    mask = torch.cat([torch.zeros(batch_size, 1, device=device), mask], dim=1)

    return mask

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
    def __init__(self, in_channels=1, emb_size=64, patch_size=10, num_layers=2, nhead=4):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_size = emb_size
        
        # Encoder components
        self.encoder_embed = PatchEmbedding1D(self.in_channels, self.emb_size, patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder components
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.emb_size, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Reconstruct layer: Projects token dim (emb_size) back to patch dim (C * P)
        self.reconstruct = nn.Linear(self.emb_size, self.in_channels * patch_size)

    def forward(self, x, mask_ratio=0.4):
        # x: [B, C, T]
        B, C, T = x.shape
        num_patches = T // self.patch_size
        
        # Prepare Patches and Ground Truth
        # Use proj to get the patches before embedding/adding CLS/POS (for ground truth)
        patches = self.encoder_embed(x)  # [B, num_patches+1, emb_size]
        
        # Generate and Apply Mask
        # Generate a mask for the patches (excluding the CLS token)
        mask = generate_mask(B, num_patches, mask_ratio, x.device) # [num_patches] (bool)
        keep = mask == 0    # True where we keep patches

        # num_keep = (keep[0] == True).numel()  # or num_patches - num_mask
        z_masked = torch.stack([patches[b, keep[b]] for b in range(B)], dim=0) # [B, num_keep+1, emb_size]

        memory = self.encoder(z_masked)                    # [B, num_keep+1, emb]

        # Decode
        # Target for decoder is the token sequence (no CLS token)
        original_no_cls = patches[:, 1:, :] 
        memory_no_cls = memory[:, 1:, :]                   # [B, num_keep, emb]

        # Decoder will reconstruct ALL patches (visible and masked)
        tgt = torch.zeros_like(original_no_cls)             # start with zeros
        decoded = self.decoder(tgt, memory_no_cls)                # [B, num_patches, emb]

        # Reconstruct Patches
        # patches: [B, num_patches, C * patch_size]
        predicted_patches = self.reconstruct(decoded) 
        # Reshape predicted patches to [B, num_patches, C, patch_size]
        true_patches = patch_proj(x, self.patch_size)  # [B, num_patches, C * patch_size]

        # Calculate Loss ONLY on Masked Segments
        keep_no_cls = mask[:,1:] == 1       # [B, num_patches] (no CLS token)
        # Get the predicted patches corresponding to the masked tokens
        predicted_masked = torch.stack([predicted_patches[b, keep_no_cls[b]] for b in range(B)], dim=0) # [B, num_keep, C * patch_size]   
        true_masked = torch.stack([true_patches[b, keep_no_cls[b]] for b in range(B)], dim=0) # [B, num_keep, C * patch_size]   
        # Calculate Mean Squared Error (MSE) loss only on the masked samples
        # The loss is a raw signal reconstruction loss, but restricted to the masked regions.
        loss = F.mse_loss(predicted_masked, true_masked)
        # For evaluation/visualization, unflatten the predicted patches back to a signal
        # recon_signal = patch_proj_inv(predicted_patches, C, self.patch_size)
        
        return {'recon':predicted_patches, 'loss': loss,'embedding': memory[:, 0, :], 'mask': mask} # Return reconstruction, the SSL loss, and CLS token





