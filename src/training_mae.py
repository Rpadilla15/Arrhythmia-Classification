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
    json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'leads_MLII_V1_tst0.25_rd42.json')
    output_dir = os.path.join(DATA_DIR, 'output', 'mae_beat')


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
        patch_size=16,
        num_layers=2,
        nhead=4,
        max_len=256
    )
    
    # Train
    history = train_mae(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=1,
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

    # Plot training curves    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].plot(history['learning_rate'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print("Training curves saved to training_curves.png")
    plt.show()
