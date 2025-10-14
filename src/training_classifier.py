from pathlib import Path
import os
from model.transformer import TransformerAutoencoder
from model.MLP_head import MLP_head
from data_handling.dataset import ECGDataset, ECGAugmentation
import matplotlib.pyplot as plt
import torch
from model.training_utils import train, FocalLoss, CBLoss
import argparse
import numpy as np

BASE_DIR = Path(__file__).resolve()
ROOT_DIR = BASE_DIR.parent.parent.parent
DATA_DIR =  os.path.join(ROOT_DIR, 'Data')

def parse_args():
    parser = argparse.ArgumentParser(description='Train Classifier')
    parser.add_argument('--exp_name', type=str, required=True,
                       help='Experiment name')
    parser.add_argument('--save_dir', type=str, default=os.path.join(DATA_DIR, 'output', 'classif_model'),
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


    preprocessed_data_dir = os.path.join(DATA_DIR, 'input', 'preprocessed_data',"fs250_bp0.5-45Hz_oneHot_01")
    
    json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'MLII_V1_strat_60_20_20_rd42.json')


    encoder_path = os.path.join(DATA_DIR, 'output','mae_beat', 'checkpoint_epoch_100.pt')


    train_transfrom = ECGAugmentation(preset='clsf')
    
    train_dataset = ECGDataset(
    data_dir=preprocessed_data_dir,
    transform=train_transfrom,
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
    emb_size=128

    encoder = TransformerAutoencoder(
        in_channels=2,
        emb_size=emb_size,
        patch_size=16,
        num_layers=2,
        nhead=4,
        max_len=256
    )

    # # Load pretrained weights
    # encoder_saved = torch.load(encoder_path, 
    #     map_location='cpu',  # Safest default device to map to
    #     weights_only=False   # Overrides the security check
    # )
    encoder.load_state_dict(encoder_saved["model_state_dict"])
    # encoder.eval()
    # # 2. Freeze all parameters
    # for param in encoder.parameters():
    #     param.requires_grad = False 

    classifier = MLP_head(encoder=encoder, emb_size=emb_size, num_classes=2)
    
    #loss
    
    samples_per_class = np.array([0,0]).astype(np.float64)
    for i in train_dataset.labels:
        samples_per_class += i.sum(axis=0)
    print("Samples per class:", samples_per_class)
    alpha = 1.0 / (samples_per_class + 1e-6)
    alpha = alpha / alpha.sum() * len(samples_per_class)  # Normalize to num_classes
    alpha = torch.tensor(alpha, dtype=torch.float32)

    criterion = FocalLoss(alpha=alpha)
    # criterion = CBLoss(samples_per_class, beta=0.9999, loss_type='focal')

    # Train
    history = train(
        model=classifier,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=100,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=0.05,
        mask_ratio=0.60,
        warmup_epochs=10,
        device='cuda',
        save_dir=output_dir,
        patience=20,
        num_workers=4,
        criterion=criterion
        # ,strategy='mixup'
    )

    
    
