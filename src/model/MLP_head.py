import torch
import torch.nn as nn

class ECGClassifier(nn.Module):
    def __init__(self, encoder, num_classes=5):
        super().__init__()
        self.encoder = encoder
        self.pool = nn.AdaptiveAvgPool1d(1)  # handles variable length
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        z = self.encoder(x)
        z = self.pool(z).squeeze(-1)
        return self.fc(z)
