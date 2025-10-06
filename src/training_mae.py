from pathlib import Path
import os
from model.transformer import TransformerAutoencoder
from data_handling.dataset import ECGDataset
import matplotlib.pyplot as plt
from model.training_utils import train_mae





if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve()
    ROOT_DIR = BASE_DIR.parent.parent.parent
    DATA_DIR =  os.path.join(ROOT_DIR, 'Data')


    preprocessed_data_dir = os.path.join(DATA_DIR, 'input', 'preprocessed_data',"fs250_bp0.5-45Hz")
    # json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'leads_MLII_V1_tst0.25_rd42.json')
    json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', '1.json')

    output_dir = os.path.join(DATA_DIR, 'output', 'test')


    # Example: signals [N, 1, 2048]
    train_dataset = ECGDataset(
    data_dir=preprocessed_data_dir,
    window_size=256,
    json_file_path=json_split_path,      
    mode="ssl")

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
        patch_size=128,
        num_layers=2,
        nhead=4,
        max_len=256
    )
    
    # Train
    history = train_mae(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=20,
        batch_size=256,
        learning_rate=1e-3,
        weight_decay=0.05,
        mask_ratio=0.60,
        warmup_epochs=10,
        device='cuda',
        save_dir=output_dir,
        patience=20,
        num_workers=4
    )

    
    
