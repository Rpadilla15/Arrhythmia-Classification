import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt, resample
import os
import pandas as pd
import json

def build_patient_sensor_index(csv_dir, out_file):
    """
    Reads all CSV files in a directory and builds an index of which leads each patient has.
    
    Args:
        csv_dir: path to directory with patient CSV files
        out_file: where to save the index (JSON)
    """
    index = {}

    for fname in os.listdir(csv_dir):
        if not fname.endswith(".csv"):
            continue

        rec_id = os.path.splitext(fname)[0]
        df = pd.read_csv(os.path.join(csv_dir, fname), nrows=1)  # read header only
        df.columns = df.columns.str.strip().str.strip("'").str.strip('"')
        leads = [col for col in df.columns if col.lower() not in ["sample #"]]
        index[rec_id] = {"leads": leads}

    # Save as JSON
    with open(out_file, "w") as f:
        json.dump(index, f, indent=2)

    print(f"Index saved to {out_file}")
    return index


class ECGPreprocessor:
    def __init__(self, target_fs=None, bandpass=(0.5, 45), fs=360):
        """
        Args:
            fs: original sampling frequency
            target_fs: resample frequency (if None, keep fs)
            bandpass: (low, high) cutoff frequencies in Hz
        """
        self.fs = fs
        self.target_fs = target_fs
        self.bandpass = bandpass

        # Design bandpass filter
        nyq = 0.5 * fs
        b, a = butter(2, [bandpass[0]/nyq, bandpass[1]/nyq], btype='band')
        self.b, self.a = b, a

    def filter_signal(self, sig):
        return filtfilt(self.b, self.a, sig)

    def normalize(self, sig):
        return (sig - np.mean(sig)) / (np.std(sig) + 1e-6)

    def resample_signal(self, sig):
        if self.target_fs and self.target_fs != self.fs:
            n_samples = int(len(sig) * self.target_fs / self.fs)
            sig = resample(sig, n_samples)
        return sig

    def signal_process(self, sig):
        sig = self.filter_signal(sig)
        sig = self.normalize(sig)
        sig = self.resample_signal(sig)
        return sig
    
    def process(self, sig_df, ann_df,mapping):
        # Map labels to AMII groups
        ann, labels = map_annotations(ann_df, mapping)

        # Process the signal for each lead 
        signals = {}
        for lead in sig_df.columns[1:]:  # skip "sample #"
            sig = sig_df[lead].values
            sig_proc = self.signal_process(sig)
            signals[lead] = sig_proc
        if self.target_fs and self.target_fs != self.fs:
            # Adjust annotations
            ann = (ann * self.target_fs / self.fs).astype(int)
        
        annotations = {
            "samples": ann,
            "labels": labels
        }

        return signals, annotations


def process_record(csv_file, ann_file, out_dir, mapping, preprocessor):
    # Load ECG CSV
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.strip("'").str.strip('"')

    # Load annotations
    ann_df = pd.read_csv(ann_file, sep='\s+', index_col=False, skiprows=1,
                         names=["Time", "Sample", "Type", "Sub", "Chan", "Num", "Aux"], quoting=3)
    
    
    
    signals, annotations = preprocessor.process(df,ann_df,mapping)

    # Save preprocessed signals and annotations
    record = {
        "signals": signals,
        "annotations": annotations,
    }

    rec_id = os.path.splitext(os.path.basename(csv_file))[0]
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, f"{rec_id}.npy"), record)


def map_annotations(ann_df, mapping):
    """Filter and map annotations to arrhytmia groups."""
    valid = []
    labels = []
    for i, row in ann_df.iterrows():
        raw_label = row["Type"]
        if raw_label in mapping:
            valid.append(row["Sample"])
            labels.append(mapping[raw_label])
    return np.array(valid), labels

