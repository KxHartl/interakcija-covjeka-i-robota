import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

def calculate_velocity(positions, time_step=0.01):
    """
    Izračunava brzinu iz pozicija izvršnog člana.
    
    Args:
        positions: DataFrame s kolonama x, y, z pozicija
        time_step: Vremenski korak između mjerenja (default 0.01s)
    
    Returns:
        Niz vrijednosti brzina
    """
    # Izračunaj razliku pozicija između uzastopnih mjerenja
    dx = positions['x'].diff()
    dy = positions['y'].diff()
    dz = positions['z'].diff()
    
    # Izračunaj euklidsku udaljenost
    distance = np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Brzina = udaljenost / vrijeme
    velocity = distance / time_step
    
    return velocity

def calculate_force_magnitude(forces):
    """
    Izračunava magnitudu sile iz komponenti fx, fy, fz.
    
    Args:
        forces: DataFrame s kolonama fx, fy, fz
    
    Returns:
        Niz vrijednosti magnitud sila
    """
    force_magnitude = np.sqrt(forces['fx']**2 + forces['fy']**2 + forces['fz']**2)
    return force_magnitude

def plot_data(csv_file, output_dir='plots'):
    """
    Učitava CSV datoteku i crta graf s vremenom, brzinom i silom.
    
    Args:
        csv_file: Put do CSV datoteke
        output_dir: Direktorij za spremanje grafova
    """
    # Učitaj podatke
    try:
        data = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Greška pri učitavanju datoteke {csv_file}: {e}")
        return
    
    # Provjeri potrebne kolone
    required_cols = ['x', 'y', 'z', 'fx', 'fy', 'fz']
    if not all(col in data.columns for col in required_cols):
        print(f"Datoteka {csv_file} ne sadrži sve potrebne kolone: {required_cols}")
        return
    
    # Izračunaj brzinu
    velocity = calculate_velocity(data[['x', 'y', 'z']])
    
    # Izračunaj magnitudu sile
    force = calculate_force_magnitude(data[['fx', 'fy', 'fz']])
    
    # Kreiraj vremensku os (pretpostavljamo mjerenja svakih 0.01s)
    time = np.arange(len(data)) * 0.01
    
    # Kreiraj graf
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Lijeva y-os za brzinu
    color = 'tab:blue'
    ax1.set_xlabel('Vrijeme [s]', fontsize=12)
    ax1.set_ylabel('Brzina [m/s]', color=color, fontsize=12)
    ax1.plot(time, velocity, color=color, linewidth=2, label='Brzina')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Desna y-os za silu
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Sila [N]', color=color, fontsize=12)
    ax2.plot(time, force, color=color, linewidth=2, label='Sila')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Naslov
    filename = os.path.basename(csv_file)
    plt.title(f'Originalni podaci: {filename}', fontsize=14, fontweight='bold')
    
    # Legenda
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # Spremi graf
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_plot.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Graf spremljen: {output_file}")
    
    plt.close()

def plot_all_data(data_dir='../../datasets', output_dir='plots'):
    """
    Plota sve CSV datoteke iz zadanog direktorija.
    
    Args:
        data_dir: Direktorij s CSV datotekama
        output_dir: Direktorij za spremanje grafova
    """
    # Pronađi sve CSV datoteke
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
    # Primjer upotrebe
    # plot_data('../../datasets/norsic/norsic_brza01.csv')  # Za plotanje jedne datoteke
    plot_all_data('../../datasets', 'plots')  # Za plotanje svih datoteka
