from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

def train_autoencoder(
    model, train_loader,  save_path, val_loader=None, num_epochs=10, lr=1e-3, device="cpu", lambda_freq=0.3
):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    freq_loss = FrequencyLoss()
 
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

                mse = criterion(recon, label)
                freq = freq_loss(recon, label)
                loss = mse + lambda_freq * freq

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
                        
                        mse = criterion(recon, label)
                        freq = freq_loss(recon, label)
                        loss = mse + lambda_freq * freq

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


import torch
import torch.nn as nn

class FrequencyLoss(nn.Module):
    """
    Implements a Frequency Loss (Spectral Loss) based on the Mean Squared Error
    of the magnitude spectrum derived from the Fast Fourier Transform (FFT).

    This loss helps prevent 'blurry' or overly smooth reconstructions by penalizing
    differences in the frequency domain, forcing the model to reproduce sharp
    features (high frequencies) correctly.

    The loss is calculated only on the positive frequency components (the first
    half of the spectrum, which is sufficient for real-valued signals).
    """

    def __init__(self, reduction='mean'):
        """
        Initializes the FrequencyLoss module.

        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=reduction)

    def forward(self, reconstructed_signal: torch.Tensor, target_signal: torch.Tensor) -> torch.Tensor:
        """
        Calculates the frequency loss between the reconstructed and target signals.

        Args:
            reconstructed_signal (torch.Tensor): The model's output signal.
                                                  Shape: (Batch, Channels, Length)
            target_signal (torch.Tensor): The ground-truth signal.
                                          Shape: (Batch, Channels, Length)

        Returns:
            torch.Tensor: The calculated frequency loss.
        """
        if reconstructed_signal.shape != target_signal.shape:
            raise ValueError(f"Input tensors must have the same shape. Got {reconstructed_signal.shape} and {target_signal.shape}")

        # 1. Apply Fast Fourier Transform (FFT)
        # torch.fft.fft works on the last dimension by default (the signal length)
        fft_reconstructed = torch.fft.fft(reconstructed_signal)
        fft_target = torch.fft.fft(target_signal)

        # 2. Extract the Magnitude Spectrum
        # torch.abs() computes the magnitude of the complex numbers.
        magnitude_reconstructed = torch.abs(fft_reconstructed)
        magnitude_target = torch.abs(fft_target)

        # 3. Focus on the positive frequency components (the first half)
        # For real-valued signals, the spectrum is symmetric. We only need the first half.
        # This saves computation and avoids redundancy.
        signal_length = reconstructed_signal.size(-1)
        # Only take up to the Nyquist frequency (plus one for the DC component)
        half_length = signal_length // 2 + 1
        
        magnitude_reconstructed_half = magnitude_reconstructed[..., :half_length]
        magnitude_target_half = magnitude_target[..., :half_length]

        # 4. Calculate MSE (L2) loss on the magnitude spectra
        # This penalizes differences in the overall energy distribution across frequencies.
        loss = self.mse_loss(magnitude_reconstructed_half, magnitude_target_half)

        return loss

