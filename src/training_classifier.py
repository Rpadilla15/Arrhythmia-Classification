from pathlib import Path
import os
from model.transformer import TransformerAutoencoder
from model.MLP_head import MLP_head
from data_handling.dataset import ECGDataset
import matplotlib.pyplot as plt
import torch
from model.training_utils import train_mae





if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve()
    ROOT_DIR = BASE_DIR.parent.parent.parent
    DATA_DIR =  os.path.join(ROOT_DIR, 'Data')


    preprocessed_data_dir = os.path.join(DATA_DIR, 'input', 'preprocessed_data',"fs250_bp0.5-45Hz_oneHot")
    # json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'MLII_V1_strat_60_20_20_rd42.json')
    json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', '1.json')

    output_dir = os.path.join(DATA_DIR, 'output', 'classif_model', 'test')


    # Example: signals [N, 1, 2048]
    train_dataset = ECGDataset(
    data_dir=preprocessed_data_dir,
    window_size=256,
    json_file_path=json_split_path,      
    mode="clsf")

    val_dataset = ECGDataset(
    data_dir=preprocessed_data_dir,
    window_size=256,
    json_file_path=json_split_path,      
    mode="clsf",
    split="val"
    )
    
    # Initialize model
    encoder = TransformerAutoencoder(
        in_channels=2,
        emb_size=64,
        patch_size=16,
        num_layers=2,
        nhead=4,
        max_len=256
    )

    classifier = MLP_head(encoder=encoder, num_classes=5)
    
    # Test
    samples, labels= next(iter(train_dataset))
    classif = classifier(samples.unsqueeze(0))
    print(classif)

    
    
