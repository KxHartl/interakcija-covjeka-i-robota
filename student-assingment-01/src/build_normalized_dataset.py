import os
import pandas as pd
import numpy as np
from feature_calculation import calculate_velocity_from_xyz, calculate_force_magnitude_df

def make_normalized_dataset(output_path='datasets/normalized/all_measurements.csv'):
    rows = []
    for root, dirs, files in os.walk('datasets'):
        if root.endswith('-filtered'):
            label_folder = os.path.basename(root)
            if 'brzo' in label_folder:
                klasa = 1
            elif 'sporo' in label_folder:
                klasa = 0
            else:
                continue
            prezime = os.path.basename(os.path.dirname(root)).split('-')[0]
            for f in files:
                if not f.endswith('.csv') or f.startswith('.'):
                    continue
                csv_file = os.path.join(root, f)
                df = pd.read_csv(csv_file)
                if all(col in df.columns for col in ['x', 'y', 'z', 'fx', 'fy', 'fz']):
                    velocity = calculate_velocity_from_xyz(df)
                    force = calculate_force_magnitude_df(df)
                    avg_v = np.nanmean(velocity)
                    avg_f = np.nanmean(force)
                    rows.append([prezime, klasa, avg_v, avg_f])
    # Spremi kao CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(rows, columns=['prezime', 'klasa', 'brzina', 'sila']).to_csv(output_path, index=False)
    print(f"Saved normalized dataset to {output_path}")

if __name__ == "__main__":
    make_normalized_dataset()