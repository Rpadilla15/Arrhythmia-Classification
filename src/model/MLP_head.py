import torch
import torch.nn as nn

class MLP_head(nn.Module):
    def __init__(self, encoder, emb_size, num_classes=5, pooling='cls'):
        """
        MLP classification head for Transformer encoder.
        Args:
            encoder (nn.Module): Pretrained Transformer encoder.
            emb_size (int): Embedding size from the encoder.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.pooling = pooling
        self.dropout = nn.Dropout(0.2)
        self.encoder = encoder

        imput_size = emb_size  # 'cls'
        if pooling == 'combined':
            imput_size = emb_size*2
        # self.norm = nn.LayerNorm(imput_size)
        self.fc1 = nn.Linear(imput_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.relu  = nn.ReLU()
    
    def forward(self, x):
        z, m = self.encoder.encode(x)
            # z: [B, emb_size] (CLS token)
            # m: [B, n_patches, emb_size] (patch tokens)
        if self.pooling in ['mean', 'combined']:
            m = m.mean(dim=1)  # Mean pooling
            if self.pooling == 'combined':
                z = torch.cat((z, m), dim=1)  # Combine CLS and mean
            else:
                z = m

        # z = self.norm(z)
        z = self.dropout(self.relu(self.fc1(z)))
        z = self.dropout(self.relu(self.fc2(z)))
        return self.fc3(z)
