from pathlib import Path
import os
from model.transformer import TransformerAutoencoder
from data_handling.dataset_extended import ECGDataset
from data_handling.dataset import ECGAugmentation

import matplotlib.pyplot as plt
from model.training_utils import train
import argparse
from datetime import datetime

BASE_DIR = Path(__file__).resolve()
ROOT_DIR = BASE_DIR.parent.parent.parent
DATA_DIR =  os.path.join(ROOT_DIR, 'Data')

def parse_args():
    parser = argparse.ArgumentParser(description='Train ECG MAE')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Experiment name')
    parser.add_argument('--save_dir', type=str, default=os.path.join(DATA_DIR, 'output', 'encoder'),
                       help='Base directory for checkpoints')

    return parser.parse_args()

def setup_experiment(args):
    """Setup experiment directory based on args."""
    exp_dir = os.path.join(args.save_dir, args.exp_name)
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir, exist_ok=True)
        print(f"✓ Created: {exp_dir}\n")
        return exp_dir
    
    # Ask user
    print(f"\n⚠️  Directory '{exp_dir}' already exists!")
    response = input("Overwrite? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("Experiment cancelled.")
        exit(0)
    
    print(f"Continuing with: {exp_dir}\n")
    return exp_dir


if __name__ == "__main__":
    args = parse_args()
    output_dir = setup_experiment(args)


    preprocessed_data_dir = os.path.join(DATA_DIR, 'input', 'preprocessed_data',"fs250_bp0.5-45Hz_oneHot")
    
    json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'MLII_V1_strat_60_20_20_rd42.json')

 


    train_transfrom = ECGAugmentation(preset='ssl')
    
    # --- Load dataset ---
    train_dataset = ECGDataset(
    data_dir=preprocessed_data_dir,
    json_file_path=json_split_path,
    split="train",
    mode="ssl",
    window_size=256
    )

    val_dataset = ECGDataset(
    data_dir=preprocessed_data_dir,
    window_size=256,
    json_file_path=json_split_path, 
    mode="ssl",
    split="val"
    )
    
    # Initialize model
    model = TransformerAutoencoder(
        in_channels=2,
        emb_size=128,
        patch_size=16,
        num_layers=2,
        nhead=4,
        patch_norm=True
        )
    
    # Train
    history = train(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=200,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=0.05,
        mask_ratio=0.75,
        warmup_epochs=10,
        device='cuda',
        save_dir=output_dir,
        patience=35,
        num_workers=4
    )

    
    
