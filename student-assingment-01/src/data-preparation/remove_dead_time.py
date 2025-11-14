#!/usr/bin/env python3
"""
Program za uklanjanje mrtvog vremena iz mjerenja brzine robota.

Program prolazi kroz sve CSV datoteke u datasets direktorijima i uklanja
mrtvo vrijeme - podatke s brzinom ispod praga (0.05 m/s) na početku i kraju mjerenja.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from feature_calculation import calculate_velocity_from_xyz, calculate_force_magnitude_df


def calculate_velocity(df):
    """
    Funkcija delegira calculation na calculate_velocity_from_xyz iz feature_calculation.py radi dosljednosti.
    """
    return calculate_velocity_from_xyz(df)


def calculate_force_magnitude(df):
    """
    Funkcija delegira calculation na calculate_force_magnitude_df iz feature_calculation.py radi dosljednosti.
    """
    return calculate_force_magnitude_df(df)


def remove_dead_time(df, velocity_threshold=0.05):
    if len(df) == 0:
        return df
    velocity = calculate_velocity(df)
    start_idx = 0
    for i in range(len(velocity)):
        if pd.notna(velocity.iloc[i]) and velocity.iloc[i] >= velocity_threshold:
            start_idx = i
            break
    end_idx = len(df) - 1
    for i in range(len(velocity) - 1, -1, -1):
        if pd.notna(velocity.iloc[i]) and velocity.iloc[i] >= velocity_threshold:
            end_idx = i
            break
    if start_idx >= end_idx:
        return df.iloc[0:0]
    return df.iloc[start_idx:end_idx + 1].copy()


def process_csv_files(base_path, velocity_threshold=0.05, dry_run=False):
    base_path = Path(base_path)
    datasets_path = base_path / "datasets"
    if not datasets_path.exists():
        print(f"Greška: Direktorij {datasets_path} ne postoji!")
        return
    csv_files = list(datasets_path.glob("**/*-csv/*.csv"))
    if not csv_files:
        print(f"Nema pronađenih CSV datoteka u {datasets_path}")
        return
    print(f"Pronađeno {len(csv_files)} CSV datoteka za obradu\n")
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            original_len = len(df)
            required_cols = ['timestamp', 'x', 'y', 'z']
            if not all(col in df.columns for col in required_cols):
                print(f"Preskaćem {csv_file}: nema potrebne kolone (timestamp, x, y, z)")
                continue
            df_cleaned = remove_dead_time(df, velocity_threshold)
            cleaned_len = len(df_cleaned)
            removed = original_len - cleaned_len
            print(f"Datoteka: {csv_file.relative_to(base_path)}")
            print(f"  Originalno: {original_len} redova")
            print(f"  Nakon čišćenja: {cleaned_len} redova")
            print(f"  Uklonjeno: {removed} redova ({removed/original_len*100:.1f}%)\n")
            if not dry_run:
                df_cleaned.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Greška pri obradi {csv_file}: {e}\n")
    if dry_run:
        print("\nDRY RUN - Nisu zapisane promjene. Pokrenite bez --dry-run za primjenu promjena.")
    else:
        print("\nObrada završena!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Uklanjanje mrtvog vremena iz mjerenja brzine robota")
    parser.add_argument("--base-path", type=str, default="../../", help="Osnovni direktorij projekta (default: ../../)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Prag brzine u m/s za detekciju mrtvog vremena (default: 0.05)")
    parser.add_argument("--dry-run", action="store_true", help="Prikaži što bi bilo promijenjeno bez zapisivanja promjena")
    args = parser.parse_args()
    process_csv_files(args.base_path, args.threshold, args.dry_run)
