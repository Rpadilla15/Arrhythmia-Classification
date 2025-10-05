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

# --- Example Usage ---
if __name__ == '__main__':
    # Define parameters
    BATCH_SIZE = 4
    CHANNELS = 2  # MLII and V-lead
    SIGNAL_LENGTH = 500
    NUM_EPOCHS = 5

    # Instantiate the loss functions
    mse_loss = nn.MSELoss()
    freq_loss = FrequencyLoss()
    
    # Define weighting coefficients (hyperparameters to tune)
    lambda_mse = 1.0
    lambda_freq = 0.5 

    print("--- Simulating Training Step with Combined Loss ---")
    
    # Simulate a dummy batch of ground-truth signals (X)
    # A sine wave plus some noise to simulate a periodic ECG beat
    t = torch.linspace(0, 1, SIGNAL_LENGTH)
    target_signal = torch.sin(2 * torch.pi * 5 * t).unsqueeze(0).repeat(BATCH_SIZE, CHANNELS, 1)
    target_signal += torch.randn_like(target_signal) * 0.1
    
    # Simulate a model prediction (X_hat)
    # Scenario 1: 'Good' Prediction (Slightly noisy but correct shape)
    reconstructed_good = target_signal * 1.05 + torch.randn_like(target_signal) * 0.05
    
    # Scenario 2: 'Blurry' Prediction (Damped high frequencies, high MSE)
    # We apply a smooth filter to simulate a blurry/oversmoothed output
    from torch.nn.functional import avg_pool1d
    reconstructed_blurry = avg_pool1d(target_signal, kernel_size=5, stride=1, padding=2)
    
    # Calculate losses for the Blurry Prediction
    mse_blurry = mse_loss(reconstructed_blurry, target_signal)
    freq_blurry = freq_loss(reconstructed_blurry, target_signal)
    total_loss_blurry = lambda_mse * mse_blurry + lambda_freq * freq_blurry

    print(f"\n[Blurry Prediction] MSE Loss: {mse_blurry.item():.4f}")
    print(f"[Blurry Prediction] Freq Loss: {freq_blurry.item():.4f}")
    print(f"Total Loss (lambda={lambda_freq}): {total_loss_blurry.item():.4f}")
    
    # Calculate losses for the Good Prediction
    mse_good = mse_loss(reconstructed_good, target_signal)
    freq_good = freq_loss(reconstructed_good, target_signal)
    total_loss_good = lambda_mse * mse_good + lambda_freq * freq_good

    print(f"\n[Good Prediction] MSE Loss: {mse_good.item():.4f}")
    print(f"[Good Prediction] Freq Loss: {freq_good.item():.4f}")
    print(f"Total Loss (lambda={lambda_freq}): {total_loss_good.item():.4f}")

    # Interpretation: The frequency loss for the blurry signal should be significantly
    # higher than the frequency loss for the good signal, even if their MSEs are close,
    # demonstrating its power to penalize structural errors.