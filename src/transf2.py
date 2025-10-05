from model.transformer import TransformerAutoencoder
from test import train_mae # <-- import your model definitions from the file you already have
import os
from pathlib import Path
import json
from data_handling.dataset import ECGDataset

BASE_DIR = Path(__file__).resolve()
ROOT_DIR = BASE_DIR.parent.parent.parent
DATA_DIR =  os.path.join(ROOT_DIR, 'Data')


preprocessed_data_dir = os.path.join(DATA_DIR, 'input', 'preprocessed_data',"fs250_bp0.5-45Hz")
json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'leads_MLII_V1_tst0.25_rd42.json')
output_dir = os.path.join(DATA_DIR, 'output', 'mae_test.pth')


# Example: signals [N, 1, 2048]
train_dataset = ECGDataset(
data_dir=preprocessed_data_dir,
window_size=1024,
stride=512,
json_file_path=json_split_path,      
mode="sliding")

val_dataset = ECGDataset(
data_dir=preprocessed_data_dir,
window_size=1024,
stride=512,
json_file_path=json_split_path,      
mode="sliding",
split="val"
)

# encoder = ViT1DEncoder(signal_length=1024, patch_size=16, in_channels=2,
#                        embed_dim=64, depth=4, num_heads=4, mlp_ratio=2.0,
#                        dropout=0.1, max_rel_len=1024, use_cls_token=False)

# mae = MAE1D(encoder=encoder, decoder_embed_dim=64, decoder_depth=2, mask_ratio=0.75)

model = TransformerAutoencoder(in_channels=2, emb_size=64, patch_size=32, num_layers=2, nhead=4)
train_mae(model, train_dataset, val_dataset, save_dir=output_dir, batch_size=128, epochs=20)