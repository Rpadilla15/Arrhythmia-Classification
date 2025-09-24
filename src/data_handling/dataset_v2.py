import json
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

class MITBIHDataset(Dataset):
    """
    A custom PyTorch Dataset for the preprocessed MIT-BIH Arrhythmia Database.

    This dataset can operate in two modes:
    1. 'classification': Returns a fixed-size signal window around each beat's R-peak
       and its corresponding AAMI label. This is for supervised learning.
    2. 'ssl': Returns a fixed-size sliding window of the signal, without labels.
       This is for self-supervised learning.

    """
    def __init__(self, json_file_path, npy_dir, split='train', window_size=500, mode='classification', slide_step=125):
        """
        Initializes the dataset.

        Args:
            json_file_path (str): Path to the JSON file containing the train/test patient split.
            npy_dir (str): Directory where the preprocessed .npy files are stored.
            split (str): 'train' or 'test' to select the patient subset.
            window_size (int): The number of samples for each signal window.
            mode (str): 'classification' or 'ssl' (self-supervised learning).
            slide_step (int): The step size for the sliding window in 'ssl' mode.
        """
        self.npy_dir = npy_dir
        self.window_size = window_size
        self.mode = mode
        self.slide_step = slide_step
        self.split = split
        self.all_data = []

        # Load the patient split from the JSON file
        with open(json_file_path, 'r') as f:
            patient_splits = json.load(f)
        
        patient_ids = patient_splits[split]
        print(f"Loading data for {len(patient_ids)} patients from '{split}' split...")

        leads = patient_splits['leads']
        self._load_data(patient_ids, leads)

        if not self.all_data:
            raise ValueError("No data found for the specified split and directory.")

    def _load_data(self, patient_ids, leads):
        """
        Pre-loads all data from the .npy files into memory and prepares
        the list of indices for __getitem__.
        """
        # Fetch .npy files for each patient in the set
        for patient_id in patient_ids:
            file_path = os.path.join(self.npy_dir, f"{patient_id}.npy")
            if not os.path.exists(file_path):
                print(f"Warning: File not found for patient {patient_id}. Skipping.")
                continue

            try:
                data = np.load(file_path, allow_pickle=True).item()
                signals = data['signals']
                annotations = data['annotations']
                
                # Fetch signals
                signal_0 = signals[leads[0]]
                signal_1 = signals[leads[1]]
                
                # Check for NaNs and handle them
                if np.isnan(signal_0).any():
                    signal_0 = np.nan_to_num(signal_0, nan=0.0)
                if np.isnan(signal_1).any():
                    signal_1 = np.nan_to_num(signal_1, nan=0.0)

                r_peaks = annotations['samples']
                labels = annotations['labels']

                if self.mode == 'classification':
                    # Prepare data for classification (fixed window around each beat)
                    for i in range(len(r_peaks)):
                        r_peak = r_peaks[i]
                        label = labels[i]
                        
                        start = max(0, r_peak - self.window_size // 2)
                        end = start + self.window_size
                        
                        # Adjust for window going past end of signal
                        if end > len(signal_0):
                            end = len(signal_0)
                            start = end - self.window_size
                        
                        # We store signal and annotations together
                        self.all_data.append({
                            'patient_id': patient_id,
                            'signal_0': signal_0[start:end],
                            'signal_1': signal_1[start:end],
                            'label': label
                        })

                elif self.mode == 'ssl':
                    # Prepare data for SSL (sliding window)
                    total_len = len(signal_0)
                    for start in range(0, total_len - self.window_size, self.slide_step):
                        end = start + self.window_size
                        self.all_data.append({
                            'patient_id': patient_id,
                            'signal_0': signal_0[start:end],
                            'signal_1': signal_1[start:end],
                            'label': -1 # No label for SSL
                        })
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                continue

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, idx):
        data = self.all_data[idx]
        
        signal_0 = data['signal_0']
        signal_1 = data['signal_1']

        # Stack signals to create a 2-channel tensor (C, L)
        # Transpose to get (num_channels, signal_length)
        signal_tensor = torch.tensor(
            np.stack([signal_0, signal_1]),
            dtype=torch.float32
        )

        if self.mode == 'classification':
            label_tensor = torch.tensor(data['label'], dtype=torch.long)
            # Return signal, label 
            return signal_tensor, label_tensor
        else: # 'ssl' mode
            # Return signal 
            return signal_tensor

# --- Example Usage ---
# NOTE: This example requires you to create dummy files to run it.
# You can replace this with your actual file paths.

if __name__ == '__main__':
    # # 1. Simulate the file structure and data as per the user's description
    # if not os.path.exists("simulated_data/npy"):
    #     os.makedirs("simulated_data/npy")

    # # Simulate the JSON file with train/test split
    # patient_split = {
    #     "train": ["100", "101", "102"],
    #     "test": ["103"]
    # }
    # with open("simulated_data/patient_split.json", "w") as f:
    #     json.dump(patient_split, f)

    # # Simulate patient data in .npy files
    # for patient in ["100", "101", "102", "103"]:
    #     num_samples = 3600 # 10 seconds at 360 Hz
    #     num_beats = 10

    #     signals = {
    #         "MLII": np.random.randn(num_samples),
    #         "V1": np.random.randn(num_samples) if patient != "103" else np.zeros(num_samples)
    #     }
    #     if patient == "103":
    #         # Simulate a patient with a different lead
    #         signals = {
    #             "MLII": np.random.randn(num_samples),
    #             "V4": np.random.randn(num_samples)
    #         }
        
    #     annotations = {
    #         "samples": np.sort(np.random.randint(50, num_samples - 50, num_beats)),
    #         "labels": np.random.randint(0, 5, num_beats) # Simulate AAMI labels 0-4
    #     }
        
    #     data_to_save = {
    #         "signals": signals,
    #         "annotations": annotations
    #     }
    #     np.save(f"simulated_data/npy/{patient}.npy", data_to_save)

    # 2. Create and use the Dataset and DataLoader

    import os
    from pathlib import Path

    # Directories
    BASE_DIR = Path(__file__).resolve()
    ROOT_DIR = BASE_DIR.parent.parent.parent.parent
    DATA_DIR =  os.path.join(ROOT_DIR, 'Data')

    preprocessed_data_dir = os.path.join(DATA_DIR,'input','preprocessed_data', "fs250_bp0.5-45Hz")
    json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'leads_MLII_V1_tst0.25_rd42.json')

    # A. Classification Mode
    print("\n--- Running in Classification Mode ---")
    classification_dataset = MITBIHDataset(
        json_file_path=json_split_path,
        npy_dir=preprocessed_data_dir,
        split='train',
        mode='classification'
    )
    classification_loader = DataLoader(
        classification_dataset,
        batch_size=32,
        shuffle=True
    )

    print(f"Total beats in classification dataset: {len(classification_dataset)}")
    
    # Iterate through the loader to get a batch
    batch_signals, batch_labels, batch_lead_info = next(iter(classification_loader))
    print(f"Batch signals shape: {batch_signals.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    print(f"Batch lead info shape: {batch_lead_info.shape}")

    # B. SSL (Unlabeled) Mode
    print("\n--- Running in SSL Mode ---")
    ssl_dataset = MITBIHDataset(
        json_file_path=json_split_path,
        npy_dir=preprocessed_data_dir,
        split='train',
        mode='ssl'
    )
    ssl_loader = DataLoader(
        ssl_dataset,
        batch_size=32,
        shuffle=True
    )
    
    print(f"Total windows in SSL dataset: {len(ssl_dataset)}")

    # Iterate through the loader to get a batch
    batch_signals_ssl, batch_lead_info_ssl = next(iter(ssl_loader))
    print(f"SSL Batch signals shape: {batch_signals_ssl.shape}")
    print(f"SSL Batch lead info shape: {batch_lead_info_ssl.shape}")

    # C. Testing with a different split
    print("\n--- Running on Test Split ---")
    test_dataset = MITBIHDataset(
        json_file_path=json_split_path,
        npy_dir=preprocessed_data_dir,
        split='test',
        mode='classification'
    )
    print(f"Total beats in test dataset: {len(test_dataset)}")