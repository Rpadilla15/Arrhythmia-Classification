import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
import pandas as pd
import yaml
from skmultilearn.model_selection import iterative_train_test_split
from preprocess_utils import map_annotations

# Directories
BASE_DIR = Path(__file__).resolve()
ROOT_DIR = BASE_DIR.parent.parent.parent.parent
DATA_DIR =  os.path.join(ROOT_DIR, 'Data')

def check_class_imbalance(patients_ids):
    # Load the dataset
    # Load patient sensor index
    with open(os.path.join(DATA_DIR, 'patient_sensors.json'), 'r') as f:
        patient_sensors = json.load(f)

    dataset = pd.DataFrame()
    # AAMI class groups
    with open(os.path.join(DATA_DIR, 'AAMI_MAPPING.yaml'), 'r') as f:
        mapping = yaml.load(f, Loader=yaml.FullLoader)
    # Process each record
    for rec_id, info in patient_sensors.items():
        if rec_id not in patients_ids:
            continue
        ann_file = os.path.join(DATA_DIR, 'mitbih_database', f"{rec_id}annotations.txt")
        if not os.path.exists(ann_file):
            print(f"CSV file for {rec_id} not found, skipping.")
            continue
        # Load annotations
        ann_df = pd.read_csv(ann_file, sep='\s+', index_col=False, skiprows=1,
                         names=["Time", "Sample", "Type", "Sub", "Chan", "Num", "Aux"], quoting=3)
        labels = ann_df['Type'].values
        _, labels = map_annotations(ann_df, mapping)
        temp_df = pd.DataFrame({'record_id': rec_id, 'Type': labels})
        dataset = pd.concat([dataset, temp_df], ignore_index=True)
    
    # Count occurrences of each class
    class_counts = dataset.groupby(['record_id']).value_counts().unstack(fill_value=0)
 

    return class_counts
def group_patients_by_leads(patient_sensors, required_leads):
    """
    Groups patients based on the presence of required leads.
    
    Args:
        patient_sensors: dict mapping patient IDs to their available leads
        required_leads: list of leads that must be present for a patient to be included
    
    Returns:
        List of patient IDs that have all required leads
    """
    grouped_patients = []
    
    for patient_id, info in patient_sensors.items():
        leads = info["leads"]
        if all(lead in leads for lead in required_leads):
            grouped_patients.append(patient_id)
    
    return grouped_patients

if __name__ == "__main__":
    with open(os.path.join(DATA_DIR, 'patient_sensors.json'), 'r') as f:
        patient_sensors = json.load(f)

    required_leads = ["MLII", "V1"]
    grouped_patients = group_patients_by_leads(patient_sensors, required_leads)
    print(f"Patients with leads {required_leads}: {grouped_patients}")

    # Check class imbalance
    class_counts = check_class_imbalance(grouped_patients)
    assert class_counts.shape[0] == len(grouped_patients), "Mismatch in number of patients"

    X = class_counts.index.to_numpy().reshape(-1, 1) # Reshape to be a 2D array
    y = (class_counts.values > 0).astype(int) 

    # This function ensures that label combinations are well-represented in both sets.
    X_train_patients, y_train, X_test_patients, y_test = iterative_train_test_split(X, y, test_size=0.2) # Using 30% to show split with 10 samples
    X_train_patients, y_train, X_val_patients, y_val = iterative_train_test_split(X_train_patients, y_train, test_size=0.25) # Using 30% to show split with 10 samples

    # 3. View the results
    print("--- Train Patients ---")
    print(X_train_patients.flatten()) 

    # 3. View the results
    print("--- Validation Patients ---")
    print(X_val_patients.flatten()) 

    print("\n--- Test Patients ---")
    print(X_test_patients.flatten())

    print(f"\nOriginal shape: {y.shape}")
    print(f"Train labels shape: {y_train.shape}")
    print(f"Test labels shape: {y_test.shape}")

    # You can verify the distribution
    print("\n--- Label Distribution ---")
    print("Original:\n", np.sum(y, axis=0) / len(y))
    print("Train:\n", np.sum(y_train, axis=0) / len(y_train))
    print("Val:\n", np.sum(y_val, axis=0) / len(y_val))
    print("Test:\n", np.sum(y_test, axis=0) / len(y_test))

    experiment_name = f"{'_'.join(required_leads)}_strat_60_20_20_rd42.json"
    experiment_dir = os.path.join(DATA_DIR, 'input', 'experiment')
    experriment = {
        "train": X_train_patients.flatten().tolist(),
        "freq_train": (np.sum(y_train, axis=0)).flatten().tolist(),
        "invs_freq_train": (np.sum(y_train)/np.sum(y_train, axis=0)).flatten().tolist(),
        "val": X_val_patients.flatten().tolist(),
        "test": X_test_patients.flatten().tolist(),
        "leads": required_leads
    }

    # Save splits  
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, experiment_name), 'w') as f:
        json.dump(experriment, f, indent=4)








   
