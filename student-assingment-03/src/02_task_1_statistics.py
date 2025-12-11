#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zadatak 1: Statistička analiza - Ground Truth podataka

Analizira statističke karakteristike (mean, std, min, max) 
Euler kutova i translacije iz Ground Truth podataka.

Dataset: HPE-24 sa frameima 50, 150, 250, 350, 420, 508

Autor: Student
Datum: 2025
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, List
import math

# ============================================================================
# KONSTANTE
# ============================================================================

# Dostupni framea u HPE-24 datasetu
AVAILABLE_FRAMES = [50, 150, 250, 350, 420, 508]

# ============================================================================
# POMOCNE FUNKCIJE
# ============================================================================

def load_biwi_pose_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Učitava BIWI pose datoteku (rotacija i translacija)."""
    data = np.loadtxt(path)

    if data.shape == (4, 3):
        R = data[:3, :]
        T_mm = data[3, :]
    elif data.shape == (3, 3):
        R = data
        T_mm = np.zeros(3)
    else:
        raise ValueError(f"Neočekivan oblik pose datoteke: {data.shape}")

    return R.astype(np.float32), T_mm.astype(np.float32) / 1000.0  # T u metre


def rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """Pretvara rotacijsku matricu u Euler kutove (XYZ redoslijed)."""
    assert R.shape == (3, 3)

    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])   # oko X
        yaw   = math.atan2(-R[2, 0], sy)       # oko Y
        roll  = math.atan2(R[1, 0], R[0, 0])   # oko Z
    else:
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0

    return np.degrees(np.array([pitch, yaw, roll]))


# ============================================================================
# ANALIZA
# ============================================================================

class StatisticsAnalyzer:
    """Klasa za statističku analizu Ground Truth podataka."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.euler_angles = []  # Lista Euler kutova: [pitch, yaw, roll]
        self.translations = []  # Lista translacija: [x, y, z] u metrima
        self.frame_ids = []     # Liste frame ID-eva koji su uspješno učitani
        self.results = {}

    def load_frame_data(self, frame_id: int) -> Tuple[bool, Dict]:
        """Učitava podatke za jedan frame."""
        result = {
            'frame_id': frame_id,
            'loaded': False,
            'euler': None,
            'translation': None,
            'error': None
        }

        pose_path = self.data_root / f"frame_{frame_id:05d}_pose.txt"
        
        if not pose_path.exists():
            result['error'] = f"Datoteka ne postoji: {pose_path.name}"
            return False, result

        try:
            R, T = load_biwi_pose_txt(str(pose_path))
            euler = rotmat_to_euler_xyz(R)

            result['loaded'] = True
            result['euler'] = euler.tolist()
            result['translation'] = T.tolist()

            self.euler_angles.append(euler)
            self.translations.append(T)
            self.frame_ids.append(frame_id)

            return True, result
        except Exception as e:
            result['error'] = str(e)
            return False, result

    def load_frames_list(self, frame_list: List[int]) -> None:
        """Učitava podatke za listu framea."""
        print(f"\n{'='*70}")
        print(f"Statistička analiza - Ground Truth podataka")
        print(f"{'='*70}")
        print(f"\nData folder: {self.data_root}")
        print(f"Dostupni framea: {sorted(frame_list)}")
        print(f"Učitavanje podataka...\n")

        for frame_id in sorted(frame_list):
            success, result = self.load_frame_data(frame_id)
            status = "OK" if success else "FAIL"
            print(f"[Frame {frame_id:05d}] {status}", end="")
            if not success:
                print(f" - {result['error']}")
            else:
                print()

    def compute_statistics(self) -> Dict:
        """Računa statističke mjere."""
        if len(self.euler_angles) == 0:
            print("\n⚠️ Nema učitanih podataka za analizu!\n")
            return {}

        euler_array = np.array(self.euler_angles)  # Shape: (N, 3)
        trans_array = np.array(self.translations)  # Shape: (N, 3)

        stats = {
            'total_frames': len(self.frame_ids),
            'frame_ids': self.frame_ids,
            'euler_angles': {
                'pitch': {
                    'mean': float(euler_array[:, 0].mean()),
                    'std': float(euler_array[:, 0].std()),
                    'min': float(euler_array[:, 0].min()),
                    'max': float(euler_array[:, 0].max()),
                    'range': float(euler_array[:, 0].max() - euler_array[:, 0].min())
                },
                'yaw': {
                    'mean': float(euler_array[:, 1].mean()),
                    'std': float(euler_array[:, 1].std()),
                    'min': float(euler_array[:, 1].min()),
                    'max': float(euler_array[:, 1].max()),
                    'range': float(euler_array[:, 1].max() - euler_array[:, 1].min())
                },
                'roll': {
                    'mean': float(euler_array[:, 2].mean()),
                    'std': float(euler_array[:, 2].std()),
                    'min': float(euler_array[:, 2].min()),
                    'max': float(euler_array[:, 2].max()),
                    'range': float(euler_array[:, 2].max() - euler_array[:, 2].min())
                }
            },
            'translation': {
                'x': {
                    'mean': float(trans_array[:, 0].mean()),
                    'std': float(trans_array[:, 0].std()),
                    'min': float(trans_array[:, 0].min()),
                    'max': float(trans_array[:, 0].max()),
                    'range': float(trans_array[:, 0].max() - trans_array[:, 0].min())
                },
                'y': {
                    'mean': float(trans_array[:, 1].mean()),
                    'std': float(trans_array[:, 1].std()),
                    'min': float(trans_array[:, 1].min()),
                    'max': float(trans_array[:, 1].max()),
                    'range': float(trans_array[:, 1].max() - trans_array[:, 1].min())
                },
                'z': {
                    'mean': float(trans_array[:, 2].mean()),
                    'std': float(trans_array[:, 2].std()),
                    'min': float(trans_array[:, 2].min()),
                    'max': float(trans_array[:, 2].max()),
                    'range': float(trans_array[:, 2].max() - trans_array[:, 2].min())
                }
            }
        }

        self.results = stats
        return stats

    def print_statistics(self) -> None:
        """Ispisuje statističku analizu na konzolu."""
        if not self.results:
            return

        print(f"\n{'='*70}")
        print("REZULTATI - Statistička analiza Ground Truth podataka")
        print(f"{'='*70}")

        print(f"\nUkupno procesiranih framea: {self.results['total_frames']}")
        print(f"Frame ID-evi: {self.results['frame_ids']}")

        print(f"\n{'EULER KUTOVI (stepeni)':^70}")
        print("-" * 70)
        print(f"{'Osovina':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Range':>12}")
        print("-" * 70)

        for axis in ['pitch', 'yaw', 'roll']:
            data = self.results['euler_angles'][axis]
            print(f"{axis:<10} "
                  f"{data['mean']:>12.2f} "
                  f"{data['std']:>12.2f} "
                  f"{data['min']:>12.2f} "
                  f"{data['max']:>12.2f} "
                  f"{data['range']:>12.2f}")

        print(f"\n{'TRANSLACIJA (metri)':^70}")
        print("-" * 70)
        print(f"{'Osovina':<10} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12} {'Range':>12}")
        print("-" * 70)

        for axis in ['x', 'y', 'z']:
            data = self.results['translation'][axis]
            print(f"{axis:<10} "
                  f"{data['mean']:>12.4f} "
                  f"{data['std']:>12.4f} "
                  f"{data['min']:>12.4f} "
                  f"{data['max']:>12.4f} "
                  f"{data['range']:>12.4f}")

        print(f"\n{'='*70}\n")

    def save_to_json(self, output_path: Path) -> None:
        """Sprema statističku analizu u JSON datoteku."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"Statistika spremljena u: {output_path}")

    def create_visualization_data(self) -> None:
        """Priprema podatke za vizualizaciju."""
        if len(self.euler_angles) == 0:
            print("Nema podataka za vizualizaciju")
            return

        print(f"\n{'='*70}")
        print("SUMARNI IZVJEŠTAJ")
        print(f"{'='*70}\n")

        euler_array = np.array(self.euler_angles)
        trans_array = np.array(self.translations)

        print(f"Distribucija Euler kutova:")
        print(f"  Pitch (X): {euler_array[:, 0].min():.2f}° do {euler_array[:, 0].max():.2f}° (raspon: {euler_array[:, 0].max() - euler_array[:, 0].min():.2f}°)")
        print(f"  Yaw (Y):   {euler_array[:, 1].min():.2f}° do {euler_array[:, 1].max():.2f}° (raspon: {euler_array[:, 1].max() - euler_array[:, 1].min():.2f}°)")
        print(f"  Roll (Z):  {euler_array[:, 2].min():.2f}° do {euler_array[:, 2].max():.2f}° (raspon: {euler_array[:, 2].max() - euler_array[:, 2].min():.2f}°)")

        print(f"\nDistribucija translacije:")
        print(f"  X: {trans_array[:, 0].min():.4f}m do {trans_array[:, 0].max():.4f}m (raspon: {trans_array[:, 0].max() - trans_array[:, 0].min():.4f}m)")
        print(f"  Y: {trans_array[:, 1].min():.4f}m do {trans_array[:, 1].max():.4f}m (raspon: {trans_array[:, 1].max() - trans_array[:, 1].min():.4f}m)")
        print(f"  Z: {trans_array[:, 2].min():.4f}m do {trans_array[:, 2].max():.4f}m (raspon: {trans_array[:, 2].max() - trans_array[:, 2].min():.4f}m)")

        print(f"\n{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Koristi data folder HPE-24
    data_root = Path("./student-assingment-03/data/raw/hpe/24")

    if not data_root.exists():
        print(f"\n⚠️  Putanja ne postoji: {data_root}")
        print("Prilagodi putanju prema svojoj strukturi direktorija.\n")
        exit(1)

    # Kreiraj analyzer
    analyzer = StatisticsAnalyzer(data_root)

    # Učitaj podatke za sve dostupne framee
    analyzer.load_frames_list(AVAILABLE_FRAMES)

    # Računa statističke mjere
    analyzer.compute_statistics()

    # Ispiši rezultate
    analyzer.print_statistics()

    # Spremi statistiku u JSON
    output_json = Path("./results/task_1_statistics.json")
    analyzer.save_to_json(output_json)

    # Kreiraj sumarni izvještaj
    analyzer.create_visualization_data()

    print("✓ Statistička analiza je završena!")
