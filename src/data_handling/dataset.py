import os
import torch
import numpy as np
from torch.utils.data import Dataset
import json

class ECGDataset(Dataset):
    def __init__(self, data_dir, json_file_path,
                 window_size=1000, split='train', 
                 mode="ssl"): # Added masking parameters
        """
        Args:
            data_dir(str): directory with preprocessed .npy files
            window_size(int): length of window in samples
            split (str): 'train' or 'test' to select the patient subset
            json_file_path (str): Path to the JSON file containing the train/test patient split
            mode(str): "ssl" (SSL) or "clsf" (classification)
        """
        self.window_size = window_size
        self.mode = mode
        self.split = split
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
        
        half = self.window_size // 2
        for lead in self.leads:
            sig = rec["signals"][lead]
            segs.append(sig[pos-half:pos+half])

        # shape: (n_channels, window_size)
        x = torch.tensor(np.stack(segs), dtype=torch.float32)

        if self.mode == "ssl":
            return x
    
        # Classification returns label
        elif self.mode == "clsf":
            # map sample to label
            label_idx = np.where(rec["annotations"]["samples"] == pos)[0][0]
            y = rec["annotations"]["labels"][label_idx]
            return x, y


