from torch.utils.data import DataLoader
from dataset import ECGDataset
import os
from pathlib import Path
import numpy as np


# Directories
BASE_DIR = Path(__file__).resolve()
ROOT_DIR = BASE_DIR.parent.parent.parent.parent
DATA_DIR =  os.path.join(ROOT_DIR, 'Data')

preprocessed_data_dir = os.path.join(DATA_DIR, 'input', 'preprocessed_data',"fs250_bp0.5-45Hz")
json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'leads_MLII_V1_tst0.25_rd42.json')

# file_path = os.path.join(preprocessed_data_dir, "100.npy")
    
# Single lead (MLII only)
ds1 = ECGDataset(preprocessed_data_dir, json_file_path=json_split_path, window_size=500, mode="sliding")
print(next(iter(ds1)).shape)  # torch.Size([1, 500])

# Double channel (MLII + V1)
ds2 = ECGDataset(preprocessed_data_dir, window_size=500,  json_file_path=json_split_path, mode="sliding")
print(next(iter(ds2)).shape)  # torch.Size([2, 500])
