from data_handling import (
    ECGDataset
)
from model import (
    ECGAutoencoder, 
    TransformerAutoencoder,
    train_autoencoder
)

from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader


# Directories
BASE_DIR = Path(__file__).resolve()
ROOT_DIR = BASE_DIR.parent.parent.parent
DATA_DIR =  os.path.join(ROOT_DIR, 'Data')


preprocessed_data_dir = os.path.join(DATA_DIR, 'input', 'preprocessed_data',"fs250_bp0.5-45Hz")
json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'leads_MLII_V1_tst0.25_rd42.json')


# Dataset: SSL with sliding windows (e.g. 4 seconds = 1000 samples)
train_dataset = ECGDataset(
    data_dir=preprocessed_data_dir,
    window_size=1000,
    stride=500,
    json_file_path=json_split_path,      
    mode="sliding"
)

val_dataset = ECGDataset(
    data_dir=preprocessed_data_dir,
    window_size=1000,
    stride=500,
    json_file_path=json_split_path,      
    mode="sliding",
    split="val"
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)


# Model
# autoencoder = ECGAutoencoder(in_channels=2, latent_dim=64)
autoencoder = TransformerAutoencoder(in_channels=2, emb_size=32, patch_size=10, num_layers=1, nhead=2)


# Train
output_dir = os.path.join(DATA_DIR, 'output', 'ecg_autoencoder.pth')
device = "cuda" if torch.cuda.is_available() else "cpu"
autoencoder = train_autoencoder(autoencoder, train_loader=train_loader,val_loader=val_loader, num_epochs=1, lr=1e-3, device=device, save_path=output_dir)


