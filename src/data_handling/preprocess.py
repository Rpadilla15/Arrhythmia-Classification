from preprocess_utils import *
from pathlib import Path
import os
import yaml


# Directories
BASE_DIR = Path(__file__).resolve()
ROOT_DIR = BASE_DIR.parent.parent.parent.parent
DATA_DIR =  os.path.join(ROOT_DIR, 'Data')

def main(preprocessor, mapping, out_dir):
    # Create output directory if not exists
    os.makedirs(out_dir, exist_ok=True)

    # Load patient sensor index
    with open(os.path.join(DATA_DIR, 'patient_sensors.json'), 'r') as f:
        patient_sensors = json.load(f)

    # Process each record
    for rec_id, info in patient_sensors.items():
        csv_file = os.path.join(DATA_DIR, 'mitbih_database', f"{rec_id}.csv")
        ann_file = os.path.join(DATA_DIR, 'mitbih_database', f"{rec_id}annotations.txt")
        if not os.path.exists(csv_file):
            print(f"CSV file for {rec_id} not found, skipping.")
            continue

        print(f"Processing record {rec_id}...")
        process_record(csv_file, ann_file, out_dir, mapping, preprocessor)

if __name__ == "__main__":

    # Preprocess ECG data and save in experiment folder

    # Index of patients and their leads
    if not os.path.exists(os.path.join(DATA_DIR, 'patient_sensors.json')):
        build_patient_sensor_index(os.path.join(DATA_DIR, 'mitbih_database'), out_file=os.path.join(DATA_DIR, 'patient_sensors.json'))

    # Initialize preprocessor (MIT-BIH: fs=360 Hz)
    target_fs = 250 # target frequency if resampling
    bandpass=(0.5, 45) # bandpass filter range
    preproc = ECGPreprocessor(target_fs=target_fs, bandpass=bandpass)

    # Preprocess and save to
    dir_name = f"fs{target_fs}_bp{bandpass[0]}-{bandpass[1]}Hz_oneHot"
    out_dir = os.path.join(DATA_DIR, 'input', 'preprocessed_data', dir_name)

    # AAMI class groups
    with open(os.path.join(DATA_DIR, 'AAMI_MAPPING.yaml'), 'r') as f:
        arrhy_mapping = yaml.load(f, Loader=yaml.FullLoader)

    main(preproc, out_dir=out_dir, mapping=arrhy_mapping)


