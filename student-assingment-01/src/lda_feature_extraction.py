import os
import glob
import pandas as pd
import numpy as np
from feature_calculation import calculate_velocity_from_xyz, calculate_force_magnitude_df


def extract_features_and_labels(dataset_dir):
    """
    Izvlači značajke (prosječna brzina i sila) i labelu (0=sporo, 1=brzo) iz svih CSV u -filtered direktoriju.
    Pretpostavlja  ... filename sadrži 'sporo' ili 'brzo' za oznaku.
    """
    X = []
    y = []
    fnames = []
    for csv_file in glob.glob(os.path.join(dataset_dir, '**', '*.csv'), recursive=True):
        if (not os.path.basename(csv_file).startswith('.')):
            df = pd.read_csv(csv_file)
            if all(col in df.columns for col in ['x','y','z','fx','fy','fz']):
                velocity = calculate_velocity_from_xyz(df)
                force = calculate_force_magnitude_df(df)
                avg_v = np.nanmean(velocity)
                avg_f = np.nanmean(force)
                if 'brzo' in csv_file.lower():
                    y_label = 1
                elif 'sporo' in csv_file.lower():
                    y_label = 0
                else:
                    continue  # nepoznata klasa
                X.append([avg_v, avg_f])
                y.append(y_label)
                fnames.append(csv_file)
    return np.array(X), np.array(y), fnames

if __name__ == "__main__":
    # Primjer upotrebe
    # X, y, fnames = extract_features_and_labels('../../datasets/norsic_hartl/norsic-brzina-filtered')
    pass
