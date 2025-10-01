from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def train_autoencoder(
    model, train_loader,  save_path, val_loader=None, num_epochs=10, lr=1e-3, device="cpu"
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_model_wts = None

    for epoch in range(num_epochs):
        # ----------------
        # Training
        # ----------------
        model.train()
        total_train_loss = 0.0

        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit="batch"
        ) as pbar:
            for signal, label in train_loader:
                signal, label = signal.to(device), label.to(device)   # shape: [B, C, T]
                recon, _ = model(signal)

                loss = criterion(recon, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item() * signal.size(0)

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        # ----------------
        # Validation
        # ----------------
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                with tqdm(
                    total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Val]", unit="batch"
                ) as pbar:
                    for signal, label in val_loader:
                        signal, label = signal.to(device), label.to(device)   # shape: [B, C, T]
                        recon, _ = model(signal)
                        loss = criterion(recon, label)
                        total_val_loss += loss.item() * signal.size(0)

                        pbar.set_postfix(loss=f"{loss.item():.4f}")
                        pbar.update(1)

            avg_val_loss = total_val_loss / len(val_loader.dataset)

            # ----------------
            # Save best model
            # ----------------
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, save_path)
                print(f"✅ New best model saved with Val Loss: {best_val_loss:.6f}")

        # ----------------
        # Epoch Summary
        # ----------------
        if avg_val_loss is not None:
            print(
                f"Epoch {epoch+1}/{num_epochs} "
                f"- Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}"
            )
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}")

    if best_model_wts is not None:
        model.load_state_dict(best_model_wts)
        
    return model

