import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import numpy as np
from tqdm.auto import tqdm
from model.transformer import TransformerAutoencoder


# =============================
#  Dataset
# =============================

class ECGDataset(Dataset):
    def __init__(self, data_dir, json_file_path,
                 window_size=1000, stride=500, split='train', 
                 mode="sliding", mask_ratio=0.4, mask_patch_size=50): # Added masking parameters
        """
        Args:
            data_dir(str): directory with preprocessed .npy files
            window_size(int): length of window in samples
            stride(int): stride for sliding windows (sliding mode)
            split (str): 'train' or 'test' to select the patient subset
            json_file_path (str): Path to the JSON file containing the train/test patient split
            mode(str): "sliding" (SSL) or "beat" (classification)
            mask_ratio (float): The proportion of the signal patches to randomly mask (0.0 to 1.0), used in 'sliding' mode.
            mask_patch_size (int): The size of contiguous samples treated as a "patch" for masking.
        """
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
        self.split = split
        self.mask_ratio = mask_ratio         
        self.mask_patch_size = mask_patch_size 
        self.all_data = []
        self.data_dir = data_dir

        # Load the patient split from the JSON file
        with open(json_file_path, 'r') as f:
            patient_splits = json.load(f)
        
        self.leads = patient_splits['leads']
        patient_ids = patient_splits[split]
        print(f"Loading data for {len(patient_ids)} patients from '{split}' split...")

        # preload all patients
        self.records = []
        for patient_id in patient_ids:
            file_path = os.path.join(self.data_dir, f"{patient_id}.npy")
            if not os.path.exists(file_path):
                print(f"Warning: File not found for patient {patient_id}. Skipping.")
                continue
            try:
                rec = np.load(file_path, allow_pickle=True).item()
                self.records.append(rec)
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

        # build global index (patient_idx, position)
        self.index = []
        for rec_id, rec in enumerate(self.records):
            sig_len = len(rec["signals"][self.leads[0]])
            if mode == "sliding":
                for start in range(0, sig_len - window_size, stride):
                    self.index.append((rec_id, start))
            elif mode == "beat":
                for s in rec["annotations"]["samples"]:
                    if s - window_size//2 >= 0 and s + window_size//2 <= sig_len:
                        self.index.append((rec_id, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rec_id, pos = self.index[idx]
        rec = self.records[rec_id]

        # Collect all requested leads
        segs = []
        if self.mode == "sliding":
            for lead in self.leads:
                sig = rec["signals"][lead]
                segs.append(sig[pos:pos+self.window_size])
        elif self.mode == "beat":  # beat-centered
            half = self.window_size // 2
            for lead in self.leads:
                sig = rec["signals"][lead]
                segs.append(sig[pos-half:pos+half])

        # shape: (n_channels, window_size)
        x = torch.tensor(np.stack(segs), dtype=torch.float32)

        if self.mode == "sliding":
            # Random masking
            x_masked = x.clone()
            
            patch_size = self.mask_patch_size
            window_size = self.window_size
            
            # Calculate the number of patches available
            num_patches = window_size // patch_size
            
            if num_patches == 0 or self.mask_ratio == 0.0:
                # Cannot mask if window is too small or ratio is zero
                return x, x

            num_mask_patches = int(self.mask_ratio * num_patches)
            
            # Ensure at least one patch is masked if ratio > 0 and num_patches > 0
            if num_mask_patches == 0 and self.mask_ratio > 0:
                num_mask_patches = 1

            # Randomly select patch indices to mask
            # Using np.arange(num_patches) for the pool of indices
            masked_indices = np.random.choice(
                np.arange(num_patches), 
                size=num_mask_patches, 
                replace=False
            )
            
            # Apply masking (set the selected patches in y to zero)
            for i in masked_indices:
                start = i * patch_size
                end = start + patch_size
                # Mask across all channels (leads)
                x_masked[:, start:end] = 0.0 
                
            # Return the masked input (x_masked) and the original target (x)
            return  x
    
        # Classification returns label
        elif self.mode == "beat":
            # map sample to label
            label_idx = np.where(rec["annotations"]["samples"] == pos)[0][0]
            y = rec["annotations"]["labels"][label_idx]
            return x, y


# =============================
#  Training Loop
# =============================

def train_mae(
    mae_model,
    train_dataset,
    val_dataset=None,
    save_dir="./checkpoints",
    batch_size=16,
    epochs=50,
    lr=1e-4,
    warmup_epochs=5,
    num_workers=4,
    device="cuda"
):
    os.makedirs(save_dir, exist_ok=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)

    optimizer = optim.AdamW(mae_model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    mae_model.to(device)

    for epoch in range(1, epochs + 1):
        mae_model.train()
        total_loss = 0.0

        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch}/{epochs} [Train]", unit="batch"
        ) as pbar:
            for batch in train_loader:
                batch = batch.to(device)

                out = mae_model(batch)
                loss = out['loss']

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

        # Validation
        if val_loader is not None:
            mae_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                with tqdm(
                    total=len(val_loader), desc=f"Epoch {epoch}/{epochs} [Val]", unit="batch"
                ) as pbar:
                    for batch in val_loader:
                        batch = batch.to(device)
                        out = mae_model(batch)
                        loss = out['loss']
                        val_loss += loss.item()

                        pbar.set_postfix(loss=f"{loss.item():.4f}")
                        pbar.update(1)
                        
            avg_val_loss = val_loss / len(val_loader)
            print(f"Validation Loss: {avg_val_loss:.4f}")

        
        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"mae_epoch{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state": mae_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss": avg_loss,
        }, ckpt_path)

    print("Training complete. Checkpoints saved to", save_dir)


# =============================
#  Example Usage
# =============================

if __name__ == "__main__":
    from model.test import ViT1DEncoder, MAE1D  # <-- import your model definitions from the file you already have
    import os
    from pathlib import Path
    import json

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

    train_mae(model, train_dataset, val_dataset, save_dir=output_dir, batch_size=128, epochs=400)


