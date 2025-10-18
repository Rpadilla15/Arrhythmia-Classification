import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler, Subset
from sklearn.model_selection import train_test_split

class ECGDataset(Dataset):
    def __init__(self, data_dir, json_file_path,
                 window_size=1000, stride=None,
                 split='train', mode='clsf',
                 window_type='beat',  # 'beat' or 'sliding'
                 transform=None, binary=False, fs=360, variable_stride=False):
        """
        ECG dataset supporting beat-centered or sliding-window segmentation.

        Args:
            data_dir (str): directory containing .npy files.
            json_file_path (str): JSON file with patient splits & leads.
            window_size (int): segment length in samples.
            stride (int): stride (in samples) for sliding windows.
            split (str): 'train' or 'test'.
            mode (str): 'ssl' or 'clsf'.
            window_type (str): 'beat' for beat-centered windows, 'sliding' for continuous.
            transform (callable): augmentation.
            binary (bool): binary classification flag.
            fs (int): sampling frequency (Hz).
            variable_stride (bool): use variable stride from JSON if available.
        """
        self.data_dir = data_dir
        self.window_size = window_size
        self.stride = stride or window_size // 2
        self.split = split
        self.mode = mode
        self.window_type = window_type
        self.transform = transform
        self.binary = binary
        self.fs = fs
        self.variable_stride = variable_stride

        with open(json_file_path, 'r') as f:
            patient_splits = json.load(f)
        self.leads = patient_splits['leads']
        patient_ids = patient_splits[split]
        if 'stride' in patient_splits and self.variable_stride:
            stride_rule = np.array(patient_splits['stride'], dtype=np.float64)
            self.stride = (np.round(window_size-window_size*stride_rule)).astype(np.int32)

        self.records = []
        for pid in patient_ids:
            path = os.path.join(data_dir, f"{pid}.npy")
            if not os.path.exists(path):
                print(f"Missing file for patient {pid}")
                continue
            try:
                rec = np.load(path, allow_pickle=True).item()
                self.records.append(rec)
            except Exception as e:
                print(f"Error loading {pid}: {e}")

        # Build sample index
        self.index = []
        if self.window_type == 'beat':
            self._build_beat_windows()
        elif self.window_type == 'sliding':
            self._build_sliding_windows()
        else:
            raise ValueError(f"Unknown window_type: {self.window_type}")

    # -------------------------------------------------------
    def _build_beat_windows(self):
        """Beat-centered windows around each annotated beat."""
        for rid, rec in enumerate(self.records):
            sig_len = len(rec["signals"][self.leads[0]])
            for s in rec["annotations"]["samples"]:
                if s - self.window_size//2 >= 0 and s + self.window_size//2 <= sig_len:
                    self.index.append((rid, s))

    # -------------------------------------------------------
    def _build_sliding_windows(self):
        """Continuous sliding windows with labeling rules."""
        for rid, rec in enumerate(self.records):
            sig_len = len(rec["signals"][self.leads[0]])
            ann_samples = rec["annotations"]["samples"]
            ann_labels = rec["annotations"]["labels"]

            # slide over full signal
            start = 0
            while start + self.window_size <= sig_len:
                end = start + self.window_size

                # beats that fall inside window
                in_window = (ann_samples >= start) & (ann_samples < end)
                beats_in_window = ann_labels[in_window]

                if len(beats_in_window) == 0:
                    start += self.stride
                    continue

                # apply your labeling rules
                label = self._label_window(beats_in_window)

                self.index.append((rid, start, label))
                if self.variable_stride:
                    stride = int(sum(self.stride*label))
                    start += stride
                    continue
                start += self.stride

    # -------------------------------------------------------
    def _label_window(self, beats):
        """
        Apply labeling rules:
        - all normal -> normal
        - both normal+abnormal -> abnormal
        - multiple abnormal classes -> majority abnormal class
        - ties -> earliest abnormal class
        """
        beats = np.array(beats)
        # Assume [0,1,0,0,0] or 'N' for normal, others abnormal
        normal_mask = (beats == [0,1,0,0,0]) #| (beats == 'N')
        mask_1D = normal_mask.all(axis=1)
        if normal_mask.all():
            return beats[0]

        abnormal_beats = beats[~mask_1D]
        unique, counts = np.unique(abnormal_beats, return_counts=True)
        if len(unique) == 1:
            return unique[0]  # only one abnormal type
        else:
            max_count = counts.max()
            tied = unique[counts == max_count]
            # if tie, pick the first one in sequence
            for b in abnormal_beats:
                if b in tied:
                    return b

    # -------------------------------------------------------
    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.window_type == 'beat':
            rec_id, pos = self.index[idx]
            half = self.window_size // 2
            rec = self.records[rec_id]
            segs = [rec["signals"][lead][pos-half:pos+half] for lead in self.leads]
            x = torch.tensor(np.stack(segs), dtype=torch.float32)
            
            if self.transform:
                x = self.transform(x)
            
            if self.mode == 'ssl':
                return x
            
            label_idx = rec["annotations"]["sample_to_label"].get(pos)
            label = rec["annotations"]["labels"][label_idx]
            return x, torch.tensor(label)
        
        else:  # sliding
            rec_id, start, label = self.index[idx]
            rec = self.records[rec_id]
            end = start + self.window_size
            segs = [rec["signals"][lead][start:end] for lead in self.leads]
            x = torch.tensor(np.stack(segs), dtype=torch.float32)
           
            if self.transform:
                x = self.transform(x)
            
            if self.mode == 'ssl':
                return x
            
            return x, torch.tensor(label)


class ECGAugmentation:
    """Torch-compatible ECG augmentations with presets and optional time warping."""

    PRESETS = {
        "ssl": {
            "amplitude_scale_range": (0.8, 1.2),
            "baseline_shift_range": (-0.1, 0.1),
            "noise_std": 0.01,
            "time_warp_range": (0.9, 1.1),
            "time_warp_prob": 0.5,
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
            preset (str, optional): One of {'ssl', 'clsf'} for predefined settings.
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


# ------------------------------------------------------
# Stratified subsets & balancing (supports sliding windows)
# ------------------------------------------------------

from torch.utils.data import Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import matplotlib.pyplot as plt


def stratified_subset(dataset, fraction=1.0, random_state=42):
    """
    Return a stratified subset of the dataset with roughly the same label distribution.
    Works in classification mode only. Supports both beat-centered and sliding-window datasets.
    """
    base = dataset.dataset if isinstance(dataset, Subset) else dataset

    if getattr(base, "mode", None) != "clsf":
        raise ValueError("Stratified subsets only make sense for classification mode")

    # --- extract all labels ---
    labels = get_label_array(dataset)
    indices = np.arange(len(labels))

    if fraction < 1.0:
        subset_idxs, _ = train_test_split(
            indices,
            train_size=fraction,
            stratify=labels,
            random_state=random_state
        )
        dataset = Subset(dataset, subset_idxs)

    return dataset


def get_label_array(dataset):
    """
    Return a 1-D numpy array of integer class labels for every sample in `dataset`.

    Works if:
    - Dataset is beat-centered: base.index[i] = (rec_id, pos)
    - Dataset is sliding-window: base.index[i] = (rec_id, start, label)
    - Labels are scalars, strings, one-hot, or precomputed.
    Handles torch.utils.data.Subset transparently.
    """
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(base.index))

    labels = []
    for i in indices:
        entry = base.index[i]

        # --- sliding window mode ---
        if len(entry) == 3:
            _, _, lab = entry
            # Convert string or categorical label to integer ID if possible
            if isinstance(lab, (int, np.integer)):
                labels.append(int(lab))
            elif isinstance(lab, (str, bytes)):
                # convert string rhythm/beat labels to unique integer IDs
                # you can customize this mapping globally
                labels.append(hash(lab) % (10**6))  # temp hash mapping
            else:
                labels.append(int(np.argmax(np.asarray(lab))))
            continue

        # --- beat-centered mode ---
        rec_id, pos = entry
        rec = base.records[rec_id]
        samples = rec["annotations"]["samples"]
        labels_arr = rec["annotations"]["labels"]

        # find label corresponding to that sample
        idx = np.where(samples == pos)[0]
        if len(idx) == 0:
            raise ValueError(f"Sample position {pos} not found in record {rec_id}")
        lab = labels_arr[idx[0]]

        lab = np.asarray(lab)
        if lab.ndim == 0:
            labels.append(int(lab))
        else:
            labels.append(int(np.argmax(lab)))

    # Normalize label IDs if needed (e.g., hash → small range)
    labels = np.array(labels)
    # optional: compress hashed string labels to 0..K-1
    if labels.dtype.kind not in ['i', 'u']:
        labels = labels.astype(np.int64)
    unique_vals, inv = np.unique(labels, return_inverse=True)
    return inv  # returns normalized 0..num_classes-1


def make_balanced_sampler(dataset):
    """
    Create a WeightedRandomSampler for imbalanced classification datasets.
    Also returns class_weights and class_counts.
    Works for both beat-centered and sliding-window datasets.
    """
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    binary = getattr(base, "binary", False)
    if getattr(base, "mode", None) != "clsf":
        raise ValueError("Balanced sampler only for classification mode")

    labels = get_label_array(dataset)
    if labels.size == 0:
        raise ValueError("No labels found in dataset")

    if binary:
        labels = np.where(labels == 1, 1, 0)

    # class counts and weights
    class_counts = np.bincount(labels, minlength=labels.max() + 1)
    valid = class_counts > 0
    class_weights = np.zeros_like(class_counts, dtype=float)
    class_weights[valid] = 1.0 / class_counts[valid]

    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    return sampler, class_weights, class_counts


def compute_alpha_from_weights(class_weights):
    """
    Normalize class weights for Focal Loss (alpha).
    Ensures sum(alpha) = 1.
    """
    weights = torch.tensor(class_weights, dtype=torch.float32)
    alpha = weights / weights.sum()
    return alpha


def visualize_class_distributions(dataset, sampler=None, class_names=None, relative=False):
    """
    Plot original vs. effective (sampler-based) class distributions.
    Works with both sliding-window and beat-centered datasets.
    """
    labels = get_label_array(dataset)
    binary = getattr(dataset, "binary", False)
    if binary:
        labels = np.where(labels == 1, 1, 0)

    num_classes = labels.max() + 1
    orig_counts = np.bincount(labels, minlength=num_classes)
    orig_probs = orig_counts / orig_counts.sum()

    if sampler is not None:
        weights = np.array(sampler.weights, dtype=np.float64)
        weights /= weights.sum()
        eff_probs = np.zeros(num_classes)
        for c in range(num_classes):
            eff_probs[c] = weights[labels == c].sum()
        eff_counts = eff_probs * len(labels)
    else:
        eff_counts = orig_counts.copy()
        eff_probs = orig_probs.copy()

    x = np.arange(num_classes)
    width = 0.35
    plt.figure(figsize=(8, 5))
    if relative:
        plt.bar(x - width/2, orig_probs, width, label="Original", alpha=0.8)
        plt.bar(x + width/2, eff_probs, width, label="After Sampling", alpha=0.8)
    else:
        plt.bar(x - width/2, orig_counts, width, label="Original", alpha=0.8)
        plt.bar(x + width/2, eff_counts, width, label="After Sampling", alpha=0.8)

    if class_names:
        plt.xticks(x, class_names, rotation=45)
    else:
        plt.xticks(x, [f"C{i}" for i in range(num_classes)])
    plt.ylabel("Relative Frequency" if relative else "Count")
    plt.title("Original vs. Effective Class Distribution")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()

    return {
        "orig_counts": orig_counts,
        "eff_counts": eff_counts,
        "orig_probs": orig_probs,
        "eff_probs": eff_probs,
    }
