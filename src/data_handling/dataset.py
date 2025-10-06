import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F



class ECGDataset(Dataset):
    def __init__(self, data_dir, json_file_path,
                 window_size=1000, split='train',
                 mode='ssl', transform=None):
        """
        Args:
            data_dir (str): Directory containing preprocessed .npy files.
            json_file_path (str): JSON file with train/test patient splits.
            window_size (int): Segment length in samples.
            split (str): 'train' or 'test'.
            mode (str): 'ssl' (self-supervised) or 'clsf' (classification).
            transform (callable, optional): Augmentation/transform function.
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.mode = mode
        self.split = split
        self.transform = transform

        # Load patient splits and lead configuration
        with open(json_file_path, 'r') as f:
            patient_splits = json.load(f)
        self.leads = patient_splits['leads']
        patient_ids = patient_splits[split]

        print(f"Loading data for {len(patient_ids)} patients from '{split}' split...")

        # Preload all available patient recordings
        self.records = []
        for pid in patient_ids:
            path = os.path.join(self.data_dir, f"{pid}.npy")
            if not os.path.exists(path):
                print(f"Warning: missing file for patient {pid}.")
                continue
            try:
                rec = np.load(path, allow_pickle=True).item()
                self.records.append(rec)
            except Exception as e:
                print(f"Error loading {pid}: {e}")

        # Build global index of (record_id, position)
        self.index = []
        for rid, rec in enumerate(self.records):
            sig_len = len(rec["signals"][self.leads[0]])
            for s in rec["annotations"]["samples"]:
                if s - window_size // 2 >= 0 and s + window_size // 2 <= sig_len:
                    self.index.append((rid, s))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        rec_id, pos = self.index[idx]
        rec = self.records[rec_id]

        half = self.window_size // 2
        segs = [
            rec["signals"][lead][pos - half:pos + half]
            for lead in self.leads
        ]

        # shape → (n_channels, window_size)
        x = torch.tensor(np.stack(segs), dtype=torch.float32)

        # Apply transform/augmentation (now torch-compatible)
        if self.transform is not None:
            x = self.transform(x)

        if self.mode == "ssl":
            return x

        elif self.mode == "clsf":
            label_idx = np.where(rec["annotations"]["samples"] == pos)[0][0]
            y = rec["annotations"]["labels"][label_idx]
            return x, torch.tensor(y, dtype=torch.long)


class ECGAugmentation:
    """Torch-compatible ECG augmentations with presets and optional time warping."""

    PRESETS = {
        "ssl": {
            "amplitude_scale_range": (0.8, 1.2),
            "baseline_shift_range": (-0.1, 0.1),
            "noise_std": 0.02,
            "time_warp_range": (0.8, 1.2),
            "time_warp_prob": 0.7,
        },
        "clsf": {
            "amplitude_scale_range": (0.95, 1.05),
            "baseline_shift_range": (-0.05, 0.05),
            "noise_std": 0.01,
            "time_warp_range": (0.9, 1.1),
            "time_warp_prob": 0.3,
        }
    }

    def __init__(self, preset=None, **kwargs):
        """
        Args:
            preset (str, optional): One of {'ssl', 'clsf', 'domain'} for predefined settings.
            kwargs: Override specific parameters if desired.
        """
        # Start from preset defaults if provided
        config = self.PRESETS.get(preset, {}).copy()
        config.update(kwargs)  # override with user-provided values

        self.amp_range = config.get("amplitude_scale_range", (0.9, 1.1))
        self.baseline_range = config.get("baseline_shift_range", (-0.05, 0.05))
        self.noise_std = config.get("noise_std", 0.01)
        self.time_warp_range = config.get("time_warp_range", (0.9, 1.1))
        self.time_warp_prob = config.get("time_warp_prob", 0.5)

    def __call__(self, x: torch.Tensor):
        """Apply augmentation pipeline to ECG segment [n_channels, window_size]."""
        # Amplitude scaling
        scale = torch.empty(1).uniform_(*self.amp_range).item()
        x = x * scale

        # Baseline shift
        shift = torch.empty(1).uniform_(*self.baseline_range).item()
        x = x + shift

        # Time warping
        if torch.rand(1).item() < self.time_warp_prob:
            x = self._time_warp(x)

        # Add noise
        if self.noise_std > 0:
            x = x + torch.randn_like(x) * self.noise_std

        return x

    def _time_warp(self, x: torch.Tensor):
        """Apply uniform time stretch/compression using linear interpolation."""
        n_channels, window_size = x.shape
        factor = torch.empty(1).uniform_(*self.time_warp_range).item()
        new_len = int(window_size * factor)

        x = x.unsqueeze(0)  # shape: [1, n_channels, window_size]
        x_warped = F.interpolate(x, size=new_len, mode='linear', align_corners=False)

        # Crop or pad to original length
        if new_len > window_size:
            start = (new_len - window_size) // 2
            x_warped = x_warped[:, :, start:start + window_size]
        else:
            pad = (window_size - new_len) // 2
            x_warped = F.pad(x_warped, (pad, window_size - new_len - pad))

        return x_warped.squeeze(0)
