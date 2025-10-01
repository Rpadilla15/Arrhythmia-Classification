import torch
import torch.nn as nn

class ECGAutoencoder(nn.Module):
    def __init__(self, in_channels=1, latent_dim=64):
        super().__init__()
        
        # ----- Encoder -----
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, latent_dim, kernel_size=5, stride=2, padding=2),
            nn.ReLU()
        )
        
        # ----- Decoder -----
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, in_channels, kernel_size=7, stride=2, padding=3, output_padding=1)
        )
        
    def forward(self, x):
        z = self.encoder(x)         # latent representation
        out = self.decoder(z)       # reconstruction
        return out, z
