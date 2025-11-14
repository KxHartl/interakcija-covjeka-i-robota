import os
import glob
import pandas as pd
import numpy as np
from feature_calculation import calculate_velocity_from_xyz, calculate_force_magnitude_df

def extract_features_and_labels(dataset_dir):
    """
    Izvlači značajke i labelu (0=sporo, 1=brzo) iz svih CSV u datasets/prezime-prezime/prezime-label-filtered/mjerenje.csv
    Label se određuje iz parent direktorija ("brzo" ili "sporo" u imenu foldera)
    """
    X = []
    y = []
    fnames = []
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        label = os.path.basename(dirpath).lower()
        label_value = None
        if 'brzo' in label:
            label_value = 1
        elif 'sporo' in label:
            label_value = 0
        else:
            continue  # direktorij bez oznake, preskoči
        for file in filenames:
            if not (file.endswith('.csv') and not file.startswith('.')):
                continue
            csv_file = os.path.join(dirpath, file)
            df = pd.read_csv(csv_file)
            if all(col in df.columns for col in ['x','y','z','fx','fy','fz']):
                velocity = calculate_velocity_from_xyz(df)
                force = calculate_force_magnitude_df(df)
                avg_v = np.nanmean(velocity)
                avg_f = np.nanmean(force)
                X.append([avg_v, avg_f])
                y.append(label_value)
                fnames.append(csv_file)
    return np.array(X), np.array(y), fnames

if __name__ == "__main__":
    # Primjer upotrebe
    # X, y, fnames = extract_features_and_labels('../../datasets/')
    pass
