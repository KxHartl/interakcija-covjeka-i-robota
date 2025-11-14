"""
Data Plotting Script for Robot Movement Classification
Klasifikacija brzih i sporih kretnji robota

Ovaj program učitava podatke o kretanju robota (brzina i sila),
filtrira periode mirovanja, i prikazuje grafičke vizualizacije.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


class DataPlotter:
    """Klasa za učitavanje i vizualizaciju podataka o kretanju robota."""
    
    def __init__(self, data_dir='data/norsic'):
        """
        Inicijalizacija plotera.
        
        Args:
            data_dir (str): Direktorij s CSV datotekama
        """
        self.data_dir = Path(data_dir)
        self.movement_threshold = 0.01  # Prag za filtriranje mirovanja
        
    def load_data(self, filename):
        """
        Učitava CSV datoteku s podacima.
        
        Args:
            filename (str): Ime datoteke
            
        Returns:
            pd.DataFrame: Učitani podaci
        """
        filepath = self.data_dir / filename
        try:
            data = pd.read_csv(filepath)
            print(f"Učitano: {filename} - {len(data)} redaka")
            return data
        except Exception as e:
            print(f"Greška pri učitavanju {filename}: {e}")
            return None
    
    def filter_stationary_periods(self, data, velocity_col='velocity'):
        """
        Filtrira periode kada robot miruje.
        
        Args:
            data (pd.DataFrame): Podaci o kretanju
            velocity_col (str): Naziv stupca s brzinom
            
        Returns:
            pd.DataFrame: Filtrirani podaci
        """
        if velocity_col not in data.columns:
            print(f"Stupac {velocity_col} ne postoji u podacima")
            return data
            
        moving_mask = np.abs(data[velocity_col]) > self.movement_threshold
        filtered_data = data[moving_mask].copy()
        
        print(f"Filtrirano: {len(data) - len(filtered_data)} redaka mirovanja")
        print(f"Preostalo: {len(filtered_data)} redaka kretanja")
        
        return filtered_data
    
    def plot_single_recording(self, data, title='Kretanje robota', 
                             velocity_col='velocity', force_col='force'):
        """
        Prikazuje grafikon brzine i sile za jedno snimanje.
        
        Args:
            data (pd.DataFrame): Podaci za prikaz
            title (str): Naslov grafikona
            velocity_col (str): Naziv stupca s brzinom
            force_col (str): Naziv stupca sa silom
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Graf brzine
        if velocity_col in data.columns:
            time = np.arange(len(data))
            ax1.plot(time, data[velocity_col], 'b-', linewidth=1.5)
            ax1.set_ylabel('Brzina (m/s)', fontsize=12)
            ax1.set_title(f'{title} - Brzina', fontsize=14)
            ax1.grid(True, alpha=0.3)
            
            # Prosječna brzina
            avg_velocity = data[velocity_col].mean()
            ax1.axhline(y=avg_velocity, color='r', linestyle='--', 
                       label=f'Prosječna: {avg_velocity:.3f} m/s')
            ax1.legend()
        
        # Graf sile
        if force_col in data.columns:
            time = np.arange(len(data))
            ax2.plot(time, data[force_col], 'g-', linewidth=1.5)
            ax2.set_ylabel('Sila (N)', fontsize=12)
            ax2.set_xlabel('Vrijeme (uzorci)', fontsize=12)
            ax2.set_title(f'{title} - Sila', fontsize=14)
            ax2.grid(True, alpha=0.3)
            
            # Prosječna sila
            avg_force = data[force_col].mean()
            ax2.axhline(y=avg_force, color='r', linestyle='--',
                       label=f'Prosječna: {avg_force:.3f} N')
            ax2.legend()
        
        plt.tight_layout()
        return fig
    
    def plot_comparison(self, slow_data, fast_data, 
                       velocity_col='velocity', force_col='force'):
        """
        Usporedni prikaz brzih i sporih kretnji.
        
        Args:
            slow_data (pd.DataFrame): Podaci sporih kretnji
            fast_data (pd.DataFrame): Podaci brzih kretnji
            velocity_col (str): Naziv stupca s brzinom
            force_col (str): Naziv stupca sa silom
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Spore kretnje - brzina
        if velocity_col in slow_data.columns:
            axes[0, 0].plot(slow_data[velocity_col], 'b-', alpha=0.6)
            axes[0, 0].set_title('Spore kretnje - Brzina', fontsize=12)
            axes[0, 0].set_ylabel('Brzina (m/s)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Brze kretnje - brzina
        if velocity_col in fast_data.columns:
            axes[0, 1].plot(fast_data[velocity_col], 'r-', alpha=0.6)
            axes[0, 1].set_title('Brze kretnje - Brzina', fontsize=12)
            axes[0, 1].set_ylabel('Brzina (m/s)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Spore kretnje - sila
        if force_col in slow_data.columns:
            axes[1, 0].plot(slow_data[force_col], 'b-', alpha=0.6)
            axes[1, 0].set_title('Spore kretnje - Sila', fontsize=12)
            axes[1, 0].set_ylabel('Sila (N)')
            axes[1, 0].set_xlabel('Vrijeme (uzorci)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Brze kretnje - sila
        if force_col in fast_data.columns:
            axes[1, 1].plot(fast_data[force_col], 'r-', alpha=0.6)
            axes[1, 1].set_title('Brze kretnje - Sila', fontsize=12)
            axes[1, 1].set_ylabel('Sila (N)')
            axes[1, 1].set_xlabel('Vrijeme (uzorci)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_scatter_features(self, data_list, labels, 
                             velocity_col='velocity', force_col='force'):
        """
        Scatter plot prosječnih značajki (brzina vs sila).
        
        Args:
            data_list (list): Lista DataFrameova
            labels (list): Lista oznaka (npr. 'sporo', 'brzo')
            velocity_col (str): Naziv stupca s brzinom
            force_col (str): Naziv stupca sa silom
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (data, label) in enumerate(zip(data_list, labels)):
            if velocity_col in data.columns and force_col in data.columns:
                avg_velocity = data[velocity_col].mean()
                avg_force = data[force_col].mean()
                
                ax.scatter(avg_velocity, avg_force, 
                          c=colors[i % len(colors)], 
                          s=100, alpha=0.7, label=label)
        
        ax.set_xlabel('Prosječna brzina (m/s)', fontsize=12)
        ax.set_ylabel('Prosječna sila (N)', fontsize=12)
        ax.set_title('Scatter plot: Prosječna brzina vs Prosječna sila', 
                    fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_histograms(self, slow_data, fast_data,
                       velocity_col='velocity', force_col='force'):
        """
        Histogrami za usporedbu distribucija brzina i sila.
        
        Args:
            slow_data (pd.DataFrame): Podaci sporih kretnji
            fast_data (pd.DataFrame): Podaci brzih kretnji
            velocity_col (str): Naziv stupca s brzinom
            force_col (str): Naziv stupca sa silom
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram brzina
        if velocity_col in slow_data.columns and velocity_col in fast_data.columns:
            axes[0].hist(slow_data[velocity_col], bins=30, alpha=0.5, 
                        color='blue', label='Spore kretnje')
            axes[0].hist(fast_data[velocity_col], bins=30, alpha=0.5, 
                        color='red', label='Brze kretnje')
            axes[0].set_xlabel('Brzina (m/s)', fontsize=12)
            axes[0].set_ylabel('Frekvencija', fontsize=12)
            axes[0].set_title('Distribucija brzina', fontsize=14)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
        
        # Histogram sila
        if force_col in slow_data.columns and force_col in fast_data.columns:
            axes[1].hist(slow_data[force_col], bins=30, alpha=0.5, 
                        color='blue', label='Spore kretnje')
            axes[1].hist(fast_data[force_col], bins=30, alpha=0.5, 
                        color='red', label='Brze kretnje')
            axes[1].set_xlabel('Sila (N)', fontsize=12)
            axes[1].set_ylabel('Frekvencija', fontsize=12)
            axes[1].set_title('Distribucija sila', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def main():
    """Glavna funkcija za testiranje plotera."""
    
    # Kreiranje direktorija ako ne postoji
    os.makedirs('data/norsic', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Inicijalizacija plotera
    plotter = DataPlotter(data_dir='data/norsic')
    
    print("="*60)
    print("Program za vizualizaciju podataka o kretanju robota")
    print("="*60)
    
    # Primjer učitavanja podataka
    # NAPOMENA: Zamijeniti s pravim imenima datoteka
    example_files = [
        'slow_movement_01.csv',
        'fast_movement_01.csv'
    ]
    
    print("\nPrimjer korištenja:")
    print("1. Postavite CSV datoteke u direktorij 'data/norsic/'")
    print("2. CSV datoteke trebaju sadržavati stupce 'velocity' i 'force'")
    print("3. Pokrenite program za generiranje vizualizacija")
    
    # Demonstracija s dummy podacima ako datoteke ne postoje
    print("\nGeneriranje demo podataka...")
    
    # Demo spori pokret
    time_slow = np.linspace(0, 10, 1000)
    slow_velocity = 0.1 * np.sin(0.5 * time_slow) + np.random.normal(0, 0.01, 1000)
    slow_force = 2.0 + 0.5 * np.sin(0.3 * time_slow) + np.random.normal(0, 0.1, 1000)
    slow_df = pd.DataFrame({
        'time': time_slow,
        'velocity': slow_velocity,
        'force': slow_force
    })
    
    # Demo brzi pokret
    time_fast = np.linspace(0, 10, 1000)
    fast_velocity = 0.5 * np.sin(2.0 * time_fast) + np.random.normal(0, 0.05, 1000)
    fast_force = 5.0 + 2.0 * np.sin(1.5 * time_fast) + np.random.normal(0, 0.3, 1000)
    fast_df = pd.DataFrame({
        'time': time_fast,
        'velocity': fast_velocity,
        'force': fast_force
    })
    
    # Filtriranje podataka
    print("\nFiltriranje mirovanja...")
    slow_filtered = plotter.filter_stationary_periods(slow_df)
    fast_filtered = plotter.filter_stationary_periods(fast_df)
    
    # Generiranje grafikona
    print("\nGeneriranje vizualizacija...")
    
    # 1. Pojedinačni grafikoni
    fig1 = plotter.plot_single_recording(slow_filtered, 
                                         title='Sporo kretanje')
    fig1.savefig('figures/slow_movement_plot.png', dpi=150)
    print("Spremljeno: figures/slow_movement_plot.png")
    
    fig2 = plotter.plot_single_recording(fast_filtered, 
                                         title='Brzo kretanje')
    fig2.savefig('figures/fast_movement_plot.png', dpi=150)
    print("Spremljeno: figures/fast_movement_plot.png")
    
    # 2. Usporedni prikaz
    fig3 = plotter.plot_comparison(slow_filtered, fast_filtered)
    fig3.savefig('figures/comparison_plot.png', dpi=150)
    print("Spremljeno: figures/comparison_plot.png")
    
    # 3. Scatter plot
    fig4 = plotter.plot_scatter_features([slow_filtered, fast_filtered],
                                         ['Sporo', 'Brzo'])
    fig4.savefig('figures/scatter_plot.png', dpi=150)
    print("Spremljeno: figures/scatter_plot.png")
    
    # 4. Histogrami
    fig5 = plotter.plot_histograms(slow_filtered, fast_filtered)
    fig5.savefig('figures/histograms.png', dpi=150)
    print("Spremljeno: figures/histograms.png")
    
    print("\n" + "="*60)
    print("Vizualizacije uspješno generirane!")
    print("Provjerite direktorij 'figures/' za rezultate.")
    print("="*60)
    
    # Prikaži grafikone
    plt.show()


if __name__ == "__main__":
    main()
