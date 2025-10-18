import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, WeightedRandomSampler, Subset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


class ECGDataset(Dataset):
    def __init__(self, data_dir, json_file_path,
                 window_size=1000, split='train',
                 mode='ssl', transform=None, binary=False, important=False):
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
        self.binary = binary
        self.important = important

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
        self.labels = []  # only for classification mode
        for rid, rec in enumerate(self.records):
            sig_len = len(rec["signals"][self.leads[0]])
            self.labels.append(rec["annotations"]["labels"])
            for s, l in zip(rec["annotations"]["samples"], rec["annotations"]["labels"]):
                get_beat = True
                if self.important and ((l[0]==1) | (l[2]==1)):
                    get_beat = False

                if s - window_size // 2 >= 0 and s + window_size // 2 <= sig_len and get_beat:
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

        x = torch.tensor(np.stack(segs), dtype=torch.float32)

        if self.transform is not None:
            x = self.transform(x)

        if self.mode == "ssl":
            return x

        elif self.mode == "clsf":
            label_idx = rec["annotations"]["sample_to_label"].get(pos)
            y = rec["annotations"]["labels"][label_idx]
            if self.binary:
                binary_y = np.zeros(2)
                binary_y[int(y[1] == 1)] = 1 
                y = binary_y
            if self.important and not self.binary:
                important_y = np.zeros(3)
                if y[1] == 1:
                    important_y[0] = 1
                elif y[3] == 1:
                    important_y[1] = 1
                else:
                    important_y[2] = 1
                y = important_y
            return x, torch.tensor(y, dtype=torch.float32)


# ------------------------------------------------------
# Stratified subsets & balancing
# ------------------------------------------------------

def stratified_subset(dataset, fraction=1.0, random_state=42):
    """
    Return a stratified subset of the dataset with roughly the same label distribution.
    Works in classification mode only.
    """
    base = dataset.dataset if isinstance(dataset, Subset) else dataset

    if base.mode != 'clsf':
        raise ValueError("Stratified subsets only make sense for classification mode")

    labels = []
    for rec_id, pos in base.index:
        rec = base.records[rec_id]
        label_idx = np.where(rec["annotations"]["samples"] == pos)[0][0]
        labels.append(rec["annotations"]["labels"][label_idx])

    labels = np.array(labels)
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
    Works if labels are stored as one-hot vectors (floats) or integer scalars.
    Handles torch.utils.data.Subset transparently.
    """
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    # use subset indices if dataset is a Subset
    indices = dataset.indices if isinstance(dataset, Subset) else range(len(base.index))

    labels = []
    for i in indices:
        # base.index stores (rec_id, pos)
        rec_id, pos = base.index[i]
        rec = base.records[rec_id]
        # find label for that sample position
        label_idx = np.where(rec["annotations"]["samples"] == pos)[0][0]
        lab = rec["annotations"]["labels"][label_idx]

        # lab may be one-hot vector (np.array floats) or integer
        lab = np.asarray(lab)
        if lab.ndim == 0:           # scalar label
            labels.append(int(lab))
        else:
            # assume one-hot or probability vector -> take argmax
            labels.append(int(np.argmax(lab)))
    return np.array(labels, dtype=np.int64)


def make_balanced_sampler(dataset):
    """
    Create a WeightedRandomSampler for imbalanced classification datasets.
    Also returns class_weights for optional use in Focal Loss (alphas).
    Uses get_label_array() so works with Subset and one-hot labels.
    """
    base = dataset.dataset if isinstance(dataset, Subset) else dataset
    binary = getattr(base, 'binary', False)
    if getattr(base, "mode", None) != "clsf":
        raise ValueError("Balanced sampler only for classification mode")

    labels = get_label_array(dataset)        # 1-D int array
    if binary:
        # For binary classification, ensure labels are 0 and 1
        labels = np.where(labels == 1, 1, 0)
    # handle case where some classes may be missing in the subset
    if labels.size == 0:
        raise ValueError("No labels found in dataset")

    # bincount over the labels gives counts per class id
    class_counts = np.bincount(labels, minlength=labels.max() + 1)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights.tolist(),
        num_samples=len(sample_weights),
        replacement=True
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


import matplotlib.pyplot as plt

def visualize_class_distributions(dataset, sampler=None, class_names=None, relative=False):
    """
    Plot original vs. effective (sampler-based) class distributions.

    Args:
        dataset: ECGDataset or torch.utils.data.Subset
        sampler: WeightedRandomSampler or None
        class_names: optional list of class names to use as x-axis labels
        relative: if True, plot relative frequencies; else absolute counts
    """
    labels = get_label_array(dataset)
    binary = getattr(dataset, 'binary', False)
    if binary:
        # For binary classification, ensure labels are 0 and 1
        labels = np.where(labels == 1, 1, 0)

    num_classes = labels.max() + 1
    orig_counts = np.bincount(labels, minlength=num_classes)
    orig_probs = orig_counts / orig_counts.sum()

    if sampler is not None:
        # effective distribution after applying WeightedRandomSampler
        weights = np.array(sampler.weights, dtype=np.float64)
        weights /= weights.sum()
        eff_probs = np.zeros(num_classes)
        for c in range(num_classes):
            eff_probs[c] = weights[labels == c].sum()
        eff_counts = eff_probs * len(labels)
    else:
        eff_counts = orig_counts.copy()
        eff_probs = orig_probs.copy()

    # --- Plot ---
    x = np.arange(num_classes)
    width = 0.35

    plt.figure(figsize=(8, 5))
    if relative:
        plt.bar(x - width/2, orig_probs, width, label='Original', alpha=0.8)
        plt.bar(x + width/2, eff_probs, width, label='After Sampling', alpha=0.8)
    else:
        plt.bar(x - width/2, orig_counts, width, label='Original', alpha=0.8)
        plt.bar(x + width/2, eff_counts, width, label='After Sampling', alpha=0.8)

    if class_names:
        plt.xticks(x, class_names, rotation=45)
    else:
        plt.xticks(x, [f'C{i}' for i in range(num_classes)])
    plt.ylabel('Relative Frequency')
    plt.title('Original vs. Effective Class Distribution')
    plt.legend()
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    return {
        "orig_counts": orig_counts,
        "eff_counts": eff_counts,
        "orig_probs": orig_probs,
        "eff_probs": eff_probs
    }
