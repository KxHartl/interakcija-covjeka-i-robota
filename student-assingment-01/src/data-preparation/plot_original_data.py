import pandas as pd
import numpy as np
import os
import glob
from feature_calculation import calculate_velocity_from_xyz, calculate_force_magnitude_df

def plot_data(csv_file, output_dir='plots'):
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Greška pri učitavanju datoteke {csv_file}: {e}")
        return

    required_cols = ['x', 'y', 'z', 'fx', 'fy', 'fz']
    if not all(col in data.columns for col in required_cols):
        print(f"Datoteka {csv_file} ne sadrži sve potrebne kolone: {required_cols}")
        return

    velocity = calculate_velocity_from_xyz(data)
    force = calculate_force_magnitude_df(data)
    time = np.arange(len(data)) * 0.01

    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color = 'tab:blue'
    ax1.set_xlabel('Vrijeme [s]', fontsize=12)
    ax1.set_ylabel('Brzina [m/s]', color=color, fontsize=12)
    ax1.plot(time, velocity, color=color, linewidth=2, label='Brzina')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sila [N]', color=color, fontsize=12)
    ax2.plot(time, force, color=color, linewidth=2, label='Sila')
    ax2.tick_params(axis='y', labelcolor=color)
    filename = os.path.basename(csv_file)
    plt.title(f'Originalni podaci: {filename}', fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graf spremljen: {output_file}")
    plt.close()

def plot_all_data(data_dir='../../datasets', output_dir='plots'):
    csv_files = glob.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)
    if not csv_files:
        print(f"Nije pronađena nijedna CSV datoteka u direktoriju: {data_dir}")
        print("\nMolimo pripremite podatke u sljedećem formatu:")
        print("CSV datoteka s kolonama: x, y, z, fx, fy, fz")
        print("gdje su x, y, z pozicije izvršnog člana, a fx, fy, fz komponente sile")
        return
    print(f"Pronađeno {len(csv_files)} CSV datoteka")
    for csv_file in csv_files:
        print(f"\nObrađujem: {csv_file}")
        plot_data(csv_file, output_dir)
    print(f"\nSvi grafovi spremljeni u: {output_dir}")

if __name__ == "__main__":
    plot_all_data('datasets', 'plots')
