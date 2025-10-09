import torch
import torch.nn as nn

class MLP_head(nn.Module):
    def __init__(self, encoder, num_classes=5):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu  = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        z = self.encoder.encode(x)
        z = self.relu(self.fc1(z))
        out = self.softmax(self.fc2(z))
        return out
