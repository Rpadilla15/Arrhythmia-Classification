from pathlib import Path
import os
from model.transformer import TransformerAutoencoder
from data_handling.dataset import ECGDataset
import matplotlib.pyplot as plt
from model.training_utils import train_mae





if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve()
    ROOT_DIR = BASE_DIR.parent.parent.parent
    DATA_DIR =  os.path.join(ROOT_DIR, 'Data')


    preprocessed_data_dir = os.path.join(DATA_DIR, 'input', 'preprocessed_data',"fs250_bp0.5-45Hz")
    # json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', 'leads_MLII_V1_tst0.25_rd42.json')
    json_split_path = os.path.join(DATA_DIR, 'input', 'experiment', '1.json')

    output_dir = os.path.join(DATA_DIR, 'output', 'test')