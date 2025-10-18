import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from npeet import entropy_estimators as ee  
from model.transformer import TransformerAutoencoder
from data_handling.dataset_extended import ECGDataset
from data_handling.dataset import ECGAugmentation
from pathlib import Path
import os
from torch.utils.data import DataLoader
from data_handling.dataset_extended import stratified_subset, make_balanced_sampler, compute_alpha_from_weights,visualize_class_distributions


def evaluate(model, test_loader, augment):
    all_z1, all_z2, all_labels = [], [], []

    with torch.no_grad():
        for x, labels in tqdm(test_loader, desc="Extracting features"):
            labels = torch.argmax(labels, dim=1)
            x = x.to(device)
            x1, x2 = augment(x), augment(x)
            # --- Forward through encoder (adjust to your model)
            z1, _ = model.encode(x1)
            z2, _ = model.encode(x2)

            # Flatten if needed
            z1 = z1.mean(dim=(-1, -2)) if z1.ndim > 2 else z1
            z2 = z2.mean(dim=(-1, -2)) if z2.ndim > 2 else z2

            all_z1.append(z1.cpu())
            all_z2.append(z2.cpu())
            all_labels.append(labels)

    # Concatenate all batches
    z1_np = torch.cat(all_z1).numpy()
    z2_np = torch.cat(all_z2).numpy()
    labels_np = torch.cat(all_labels).numpy()

    # ---------------------------------------------------------------------
    # METRIC: Mutual Information between augmentations
    # ---------------------------------------------------------------------
    print("\n=== MUTUAL INFORMATION ===")
    mi_score = ee.mi(z1_np, z2_np, k=5)
    print(f"Mutual Information (augment invariance): {mi_score:.4f}")

    # Per-class MI (optional)
    unique_labels = np.unique(labels_np)
    mi_per_class = {}
    for cls in unique_labels:
        idx = labels_np == cls
        if np.sum(idx) > 20:  # skip tiny classes
            mi_per_class[int(cls)] = ee.mi(z1_np[idx], z2_np[idx])
    print("Per-class MI:", mi_per_class)



    print("\n=== SUMMARY ===")
    print(f"Mutual Information: {mi_score:.4f}")
    print("Per-class MI:", mi_per_class)

class batchAugment:
    def __init__(self, augment):
        self.augment = augment

    def __call__(self, x):        
        return torch.stack([self.augment(i) for i in x])

if __name__ == "__main__":
    # ---------------------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------------------
    BASE_DIR = Path(__file__).resolve()
    ROOT_DIR = BASE_DIR.parent.parent.parent
    DATA_DIR =  os.path.join(ROOT_DIR, 'Data')

    preprocessed_data_dir = os.path.join(DATA_DIR, 'input', 'preprocessed_data',"fs250_bp0.5-45Hz_oneHot")
    json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'MLII_V1_strat_60_20_20_rd42.json')
    encoder_path = os.path.join(DATA_DIR, 'output','mae_beat', 'checkpoint_epoch_100.pt')
    

    # Initialize model
    in_channels = 2
    emb_size=128

    signal_length = 256
    patch_size=16


    signal_augment = ECGAugmentation(preset='clsf')
    augment = batchAugment(signal_augment)

    dataset = ECGDataset(
    data_dir=preprocessed_data_dir,
    window_size=signal_length,
    json_file_path=json_split_path,      
    mode="clsf",
    split='test')

    # Use only 25% of training data (stratified)
    test_dataset = stratified_subset(dataset, fraction=1.0)

    # Balanced sampling
    sampler, class_weights, class_counts = make_balanced_sampler(test_dataset)

    test_loader = DataLoader(
        dataset,
        batch_size=256,
        sampler=sampler,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    model = TransformerAutoencoder(
        in_channels=2,
        emb_size=emb_size,
        patch_size=patch_size,
        num_layers=2,
        nhead=4,
        max_len=signal_length
        )

    # Load pretrained weights
    model_saved = torch.load(encoder_path, 
    map_location='cpu',  # Safest default device to map to
    weights_only=False   # Overrides the security check
    )
    model.load_state_dict(model_saved["model_state_dict"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model.to(device)


    # ---------------------------------------------------------------------
    # RUN EVALUATION
    # ---------------------------------------------------------------------
    evaluate(model, test_loader, augment)
