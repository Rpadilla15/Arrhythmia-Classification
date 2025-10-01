import torch
import torch.nn as nn

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
# Transformer Autoencoder
# -----------------
class TransformerAutoencoder(nn.Module):
    def __init__(self, in_channels=1, emb_size=64, patch_size=10, num_layers=2, nhead=4):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.encoder_embed = PatchEmbedding1D(self.in_channels, emb_size, patch_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=128, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.reconstruct = nn.Linear(emb_size, self.in_channels*patch_size)
        # self.reconstruct = nn.ConvTranspose1d(16, self.in_channels, kernel_size=7, stride=2, padding=3, output_padding=1)


    def forward(self, x):
        # Encode
        z = self.encoder_embed(x)        # [B, num_patches+1, emb]
        memory = self.encoder(z)         # [B, num_patches+1, emb]

        # Drop CLS token for reconstruction
        memory_no_cls = memory[:, 1:, :]

        # Decode
        tgt = torch.zeros_like(memory_no_cls)  # start with zeros
        decoded = self.decoder(tgt, memory)    # [B, num_patches, emb]

        # Reconstruct patches → signal
        patches = self.reconstruct(decoded)    # [B, num_patches, patch_size]
        recon = patches.reshape(x.size(0), self.in_channels, -1)  # [B, 1, T]
        return recon, memory[:, 0, :]  # return reconstruction + CLS embedding as latent
