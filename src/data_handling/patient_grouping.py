import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).resolve()
ROOT_DIR = BASE_DIR.parent.parent.parent.parent
DATA_DIR =  os.path.join(ROOT_DIR, 'Data')


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

    # Example: Split into train and test sets
    train_patients, test_patients = train_test_split(grouped_patients, test_size=0.25, random_state=42)
    print(f"Train patients: {train_patients}")
    print(f"Test patients: {test_patients}")

    experiment_name = f"leads_{'_'.join(required_leads)}_tst0.25_rd42.json"
    experiment_dir = os.path.join(DATA_DIR, 'input', 'experiment')
    experriment = {
        "train": train_patients,
        "test": test_patients,
        "leads": required_leads
    }

    # Save splits  
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, experiment_name), 'w') as f:
        json.dump(experriment, f, indent=4)