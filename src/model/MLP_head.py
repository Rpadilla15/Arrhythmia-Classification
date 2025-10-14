import torch
import torch.nn as nn

class MLP_head(nn.Module):
    def __init__(self, encoder, emb_size, num_classes=5):
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.encoder = encoder
        self.fc1 = nn.Linear(emb_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu  = nn.ReLU()
    
    def forward(self, x):
        z = self.encoder.encode(x)
        z = self.dropout(self.relu(self.fc1(z)))
        return self.fc2(z)
