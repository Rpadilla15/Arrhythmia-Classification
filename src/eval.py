import torch
import numpy as np
from sklearn.metrics import (
    confusion_matrix, classification_report, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from model.transformer import TransformerAutoencoder
from model.MLP_head import MLP_head
from data_handling.dataset import ECGDataset, ECGAugmentation
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

def evaluate_model(model, dataloader, device, class_names=None, plot_cm=True):
    """
    Evaluate a classification model on a given dataloader.

    Args:
        model: Trained PyTorch model that outputs logits.
        dataloader: DataLoader returning (x, y) pairs.
        device: torch.device('cuda' or 'cpu').
        class_names: Optional list of class names (len=num_classes).
        plot_cm: Whether to display confusion matrix.

    Returns:
        dict with metrics (accuracy, per-class F1, macro F1, confusion matrix)
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            # Convert one-hot labels if needed
            if y.ndim > 1:
                y = torch.argmax(y, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_per_class = f1_score(all_labels, all_preds, average=None)

    cm = confusion_matrix(all_labels, all_preds)

    print("=== Evaluation Summary ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {f1_macro:.4f}")
    print()
    print("Per-class F1:")
    if class_names:
        for name, score in zip(class_names, f1_per_class):
            print(f"  {name:10s}: {score:.4f}")
    else:
        for i, score in enumerate(f1_per_class):
            print(f"  Class {i}: {score:.4f}")

    # Optional: detailed precision/recall
    print("\nDetailed classification report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Plot confusion matrix
    if plot_cm:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm
    }

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve()
    ROOT_DIR = BASE_DIR.parent.parent.parent
    DATA_DIR =  os.path.join(ROOT_DIR, 'Data')

    preprocessed_data_dir = os.path.join(DATA_DIR, 'input', 'preprocessed_data',"fs250_bp0.5-45Hz_oneHot_01")
    
    json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'MLII_V1_strat_60_20_20_rd42.json')

    encoder_path = os.path.join(DATA_DIR, 'output','encoder','beat_centered_128_16_75', 'best_model.pt')
    classif_path = os.path.join(DATA_DIR, 'output','classif_model','full1', 'best_model.pt')


    val_dataset = ECGDataset(
    data_dir=preprocessed_data_dir,
    window_size=256,
    json_file_path=json_split_path,      
    mode="clsf",
    split="test"
    )

    test_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=4,
        pin_memory=True
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

    # Load pretrained weights
    encoder_saved = torch.load(encoder_path, 
        map_location='cpu',  # Safest default device to map to
        weights_only=False   # Overrides the security check
    )
    encoder.load_state_dict(encoder_saved["model_state_dict"])
    encoder.eval()

    classifier = MLP_head(encoder=encoder, emb_size=emb_size, num_classes=2)
     # Load pretrained weights
    classif_saved = torch.load(classif_path, 
        map_location='cpu',  # Safest default device to map to
        weights_only=False   # Overrides the security check
    )
    classifier.load_state_dict(classif_saved["model_state_dict"])
    classifier.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    # class_names = ["F", "N", "Q", "S", "V"]  # example
    class_names = ["A", "N"]  # example


    metrics = evaluate_model(classifier, test_loader, device, class_names)
    

