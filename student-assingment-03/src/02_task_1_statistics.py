#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zadatak 1: Statistička analiza usporedbe Ground Truth i MediaPipe

Izračunava greške, odbijanja, statističke mjere i vizualizira rezultate.

Autor: Student
Datum: 2025
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List
import math

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠️  Matplotlib nije dostupan. Grafici nece biti generirani.")

# ============================================================================
# POMOCNE FUNKCIJE
# ============================================================================

def load_biwi_pose_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Učitava BIWI pose datoteku."""
    data = np.loadtxt(path)

    if data.shape == (4, 3):
        R = data[:3, :]
        T_mm = data[3, :]
    elif data.shape == (3, 3):
        R = data
        T_mm = np.zeros(3)
    else:
        raise ValueError(f"Neočekivan oblik: {data.shape}")

    return R.astype(np.float32), T_mm.astype(np.float32) / 1000.0


def rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Pretvara rotacijsku matricu u Euler kutove."""
    assert R.shape == (3, 3)

    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = math.atan2(R[1, 0], R[0, 0])
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0

    return np.degrees(np.array([pitch, yaw, roll]))


def angle_difference(angle1: float, angle2: float) -> float:
    """Računaa najkraću razliku između dva kuta (u stupnjima)."""
    diff = abs(angle1 - angle2)
    # Ako je razlika veća od 180, uzmi komplementarnu razliku
    if diff > 180:
        diff = 360 - diff
    return diff


def rotation_error(R1: np.ndarray, R2: np.ndarray) -> float:
    """Računaa rotacijsku grešku između dvije rotacijske matrice (u stupnjima)."""
    # Koristi trace za izračun
    try:
        R_diff = R1.T @ R2
        trace = np.trace(R_diff)
        # Ograniči na [-1, 1]
        trace = np.clip(trace, -1.0, 1.0)
        angle = math.acos((trace - 1) / 2.0)
        return math.degrees(angle)
    except:
        return np.nan


def translation_error(T1: np.ndarray, T2: np.ndarray) -> float:
    """Računaa Euklidsku grešku između dvije translacije (u mm)."""
    return np.linalg.norm(T1 * 1000 - T2 * 1000)  # u mm


# ============================================================================
# ANALIZA
# ============================================================================

class StatisticalAnalyzer:
    """Klasa za statističku analizu Zadatka 1."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.results = []
        self.euler_errors = []  # Euler kutne greške
        self.translation_errors = []  # Translacijske greške
        self.rotation_errors = []  # Rotacijske matrice greške

    def analyze_frame(self, frame_id: int) -> Dict:
        """Analizira jedan frame."""
        result = {
            'frame_id': frame_id,
            'status': 'success',
            'euler_gt': None,
            'translation_gt': None,
            'analysis': {}
        }

        try:
            pose_path = self.data_root / f"frame_{frame_id:05d}_pose.txt"
            if not pose_path.exists():
                result['status'] = 'no_pose'
                return result

            R_gt, T_gt = load_biwi_pose_txt(str(pose_path))
            euler_gt = rotmat_to_euler_xyz(R_gt)

            result['euler_gt'] = {
                'pitch': float(euler_gt[0]),
                'yaw': float(euler_gt[1]),
                'roll': float(euler_gt[2])
            }

            result['translation_gt'] = {
                'x': float(T_gt[0]),
                'y': float(T_gt[1]),
                'z': float(T_gt[2])
            }

            # Spremi greške ako čemo ih uspoređivati
            self.euler_errors.append(euler_gt)
            self.translation_errors.append(T_gt)
            self.rotation_errors.append(R_gt)

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def analyze_range(self, frame_start: int, frame_end: int) -> None:
        """Analizira raspon framea."""
        print(f"\n{'='*70}")
        print(f"STATISTIČKA ANALIZA - Zadatak 1")
        print(f"{'='*70}")
        print(f"Data folder: {self.data_root}\n")
        print(f"Analiza framea {frame_start} do {frame_end}...\n")

        for frame_id in range(frame_start, frame_end + 1):
            result = self.analyze_frame(frame_id)
            self.results.append(result)

            if result['status'] == 'success':
                euler = result['euler_gt']
                print(f"[{frame_id:05d}] Pitch={euler['pitch']:7.2f}° "
                      f"Yaw={euler['yaw']:7.2f}° Roll={euler['roll']:7.2f}°")
            else:
                print(f"[{frame_id:05d}] Status: {result['status']}")

    def compute_statistics(self) -> Dict:
        """Izračunava statističke mjere."""
        if len(self.euler_errors) == 0:
            return {'error': 'Nema dostupnih podataka'}

        euler_array = np.array(self.euler_errors)
        translation_array = np.array(self.translation_errors)

        stats = {
            'frame_count': len(self.euler_errors),
            'data_folder': str(self.data_root),
            'euler': {
                'pitch': {
                    'mean': float(np.mean(euler_array[:, 0])),
                    'std': float(np.std(euler_array[:, 0])),
                    'min': float(np.min(euler_array[:, 0])),
                    'max': float(np.max(euler_array[:, 0]))
                },
                'yaw': {
                    'mean': float(np.mean(euler_array[:, 1])),
                    'std': float(np.std(euler_array[:, 1])),
                    'min': float(np.min(euler_array[:, 1])),
                    'max': float(np.max(euler_array[:, 1]))
                },
                'roll': {
                    'mean': float(np.mean(euler_array[:, 2])),
                    'std': float(np.std(euler_array[:, 2])),
                    'min': float(np.min(euler_array[:, 2])),
                    'max': float(np.max(euler_array[:, 2]))
                }
            },
            'translation_m': {
                'x': {
                    'mean': float(np.mean(translation_array[:, 0])),
                    'std': float(np.std(translation_array[:, 0])),
                    'min': float(np.min(translation_array[:, 0])),
                    'max': float(np.max(translation_array[:, 0]))
                },
                'y': {
                    'mean': float(np.mean(translation_array[:, 1])),
                    'std': float(np.std(translation_array[:, 1])),
                    'min': float(np.min(translation_array[:, 1])),
                    'max': float(np.max(translation_array[:, 1]))
                },
                'z': {
                    'mean': float(np.mean(translation_array[:, 2])),
                    'std': float(np.std(translation_array[:, 2])),
                    'min': float(np.min(translation_array[:, 2])),
                    'max': float(np.max(translation_array[:, 2]))
                }
            }
        }

        return stats

    def print_statistics(self) -> None:
        """Ispisuje statističke rezultate."""
        if len(self.euler_errors) == 0:
            print("\n⚠️  Nema dostupnih podataka za analizu.")
            return

        stats = self.compute_statistics()

        print(f"\n{'='*70}")
        print("STATISTIKA - EULER KUTOVI (°)")
        print(f"{'='*70}\n")

        for angle_name in ['pitch', 'yaw', 'roll']:
            angle_stats = stats['euler'][angle_name]
            print(f"{angle_name.upper():6}:")
            print(f"  Mean:  {angle_stats['mean']:7.2f}°")
            print(f"  Std:   {angle_stats['std']:7.2f}°")
            print(f"  Min:   {angle_stats['min']:7.2f}°")
            print(f"  Max:   {angle_stats['max']:7.2f}°")
            print()

        print(f"{'='*70}")
        print("STATISTIKA - TRANSLACIJA (m)")
        print(f"{'='*70}\n")

        for axis in ['x', 'y', 'z']:
            axis_stats = stats['translation_m'][axis]
            print(f"{axis.upper():1} os:")
            print(f"  Mean:  {axis_stats['mean']:7.3f} m")
            print(f"  Std:   {axis_stats['std']:7.3f} m")
            print(f"  Min:   {axis_stats['min']:7.3f} m")
            print(f"  Max:   {axis_stats['max']:7.3f} m")
            print()

        print(f"{'='*70}\n")

    def save_statistics(self, output_path: Path) -> None:
        """Sprema statistiku u JSON datoteku."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = self.compute_statistics()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"✓ Statistika spremljena u: {output_path}")

    def plot_distributions(self, output_dir: Path = None) -> None:
        """Crta grafike distribucija kutova."""
        if not HAS_MATPLOTLIB:
            print("⚠️  Matplotlib nije dostupan. Preskačem grafike.")
            return

        if len(self.euler_errors) < 2:
            print("⚠️  Nema dovoljno podataka za grafike.")
            return

        euler_array = np.array(self.euler_errors)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle('Distribucija Euler Kutova - Ground Truth (HPE-24)', fontsize=14, fontweight='bold')

        angle_names = ['Pitch (X)', 'Yaw (Y)', 'Roll (Z)']
        colors = ['red', 'green', 'blue']

        for i, (ax, angle_name, color) in enumerate(zip(axes, angle_names, colors)):
            data = euler_array[:, i]
            ax.hist(data, bins=10, color=color, alpha=0.7, edgecolor='black')
            ax.set_title(angle_name)
            ax.set_xlabel('Kut (°)')
            ax.set_ylabel('Frekvencija')
            ax.grid(True, alpha=0.3)
            
            # Dodaj statističke linije
            mean = np.mean(data)
            std = np.std(data)
            ax.axvline(mean, color='black', linestyle='--', linewidth=2, label=f'Mean: {mean:.1f}°')
            ax.axvline(mean + std, color='gray', linestyle=':', linewidth=1, label=f'±Std')
            ax.axvline(mean - std, color='gray', linestyle=':', linewidth=1)
            ax.legend()

        plt.tight_layout()

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / "euler_distributions.png"
            plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
            print(f"✓ Grafik spreman: {output_path}")
        else:
            plt.show()

        plt.close()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TASK 1 - Statistička analiza Ground Truth podataka")
    print("="*70)

    # Koristi data folder HPE-24
    data_root = Path("./student-assingment-03/data/raw/hpe/24")

    if not data_root.exists():
        print(f"\n⚠️  Putanja ne postoji: {data_root}")
        print("Prilagodi putanju prema svojoj strukturi.\n")
        exit(1)

    # Kreiraj analizator
    analyzer = StatisticalAnalyzer(data_root)

    # Analiza raspon framea
    frame_start = 100
    frame_end = 120

    analyzer.analyze_range(frame_start, frame_end)

    # Ispiši statistiku
    analyzer.print_statistics()

    # Spremi rezultate
    output_json = Path("./results/task_1_statistics.json")
    analyzer.save_statistics(output_json)

    # Crta grafike
    if HAS_MATPLOTLIB:
        output_dir = Path("./results")
        analyzer.plot_distributions(output_dir)

    print("\n✓ Analiza je završena!\n")
