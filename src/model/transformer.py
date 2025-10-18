import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_utils import *

# -----------------
# Transformer Autoencoder 
# -----------------
class TransformerAutoencoder(nn.Module):
    def __init__(self, in_channels=2, emb_size=64, patch_size=10, num_layers=2, nhead=4, max_len=5000, patch_norm=False):
        """ 1D Transformer Autoencoder with Masked Autoencoding
        Args:
            in_channels (int): Number of input channels (e.g., ECG leads)
            emb_size (int): Embedding dimension
            patch_size (int): Size of each patch (in samples)
            num_layers (int): Number of transformer layers in both encoder and decoder
            nhead (int): Number of attention heads
            max_len (int): Maximum length of the input signal (in samples)
        """
        super().__init__()
        self.patch_norm = patch_norm
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.emb_size = emb_size
        max_patches = max_len // patch_size
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.emb_size))

        # Encoder components
        self.encoder_embed = PatchEmbedding1D(self.in_channels, self.emb_size, patch_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=nhead, dim_feedforward=4*self.emb_size, batch_first=True, norm_first=True if patch_norm else False
            )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers*2)

        # Decoder components
        self.decoder_pos_embed = LearnedPositionalEmbedding(max_patches, self.emb_size)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=self.emb_size, nhead=nhead, dim_feedforward=4*self.emb_size, batch_first=True, norm_first=True if patch_norm else False
            )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        # Reconstruct layer: Projects token dim (emb_size) back to patch dim (C * P)
        self.reconstruct = nn.Linear(self.emb_size, self.in_channels * patch_size)

    
    @torch.no_grad()
    def encode(self, x):
        """Return encoder representation (no masking, no decoder)."""
        self.eval()
        patches = self.encoder_embed(x)
        memory = self.encoder(patches)
        return memory[:, 0, :],  memory[:, 1:, :]  # CLS token, memory

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
