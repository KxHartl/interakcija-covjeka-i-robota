#!/usr/bin/env python3
"""
Zadatak 1: Usporedba ground truth i MediaPipe keypoints

Autor: Krešimir Hartl
Datum: Prosinac 2025

Opis:
    Skripta učitava 5 fotografija iz PennAction dataseta, ekstrahira
    ground truth keypoints iz .mat datoteka, primjenjuje MediaPipe Pose
    detekciju i uspoređuje rezultate.
"""

import cv2
import numpy as np
import mediapipe as mp
import scipy.io as sio
import os
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple

# Inicijalizacija MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5
)

# Mapiranje PennAction keypoints (13) na MediaPipe keypoints (33)
# PennAction keypoints: head, shoulder, elbow, wrist, hip, knee, ankle (lijevo/desno)
PENNACTION_TO_MEDIAPIPE = {
    0: 0,   # head -> nose
    1: 12,  # left_shoulder -> left_shoulder
    2: 11,  # right_shoulder -> right_shoulder
    3: 14,  # left_elbow -> left_elbow
    4: 13,  # right_elbow -> right_elbow
    5: 16,  # left_wrist -> left_wrist
    6: 15,  # right_wrist -> right_wrist
    7: 24,  # left_hip -> left_hip
    8: 23,  # right_hip -> right_hip
    9: 26,  # left_knee -> left_knee
    10: 25, # right_knee -> right_knee
    11: 28, # left_ankle -> left_ankle
    12: 27  # right_ankle -> right_ankle
}

PENNACTION_KEYPOINT_NAMES = [
    "head", "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]


def load_pennaction_ground_truth(mat_file: str, frame_idx: int = 0) -> np.ndarray:
    """
    Učitava ground truth keypoints iz PennAction .mat datoteke.
    
    Args:
        mat_file: Putanja do .mat datoteke
        frame_idx: Indeks framea (default 0 za prvu sliku)
    
    Returns:
        numpy array shape (13, 2) s (x, y) koordinatama
    """
    data = sio.loadmat(mat_file)
    
    # PennAction format: x i y su odvojeni u matricama
    x_coords = data['x'][:, frame_idx]  # shape (13,)
    y_coords = data['y'][:, frame_idx]  # shape (13,)
    
    # Kombiniranje u (13, 2) format
    keypoints = np.stack([x_coords, y_coords], axis=1)
    
    return keypoints


def detect_mediapipe_keypoints(image_path: str) -> Tuple[np.ndarray, bool]:
    """
    Detektira keypoints pomoću MediaPipe Pose.
    
    Args:
        image_path: Putanja do slike
    
    Returns:
        Tuple (keypoints array shape (33, 2), success flag)
    """
    image = cv2.imread(image_path)
    if image is None:
        return None, False
    
    # Konverzija BGR -> RGB za MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detekcija pose
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return None, False
    
    # Ekstrakcija koordinata u pixel prostoru
    h, w = image.shape[:2]
    keypoints = np.zeros((33, 2))
    
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        keypoints[idx, 0] = landmark.x * w  # x koordinata u pikselima
        keypoints[idx, 1] = landmark.y * h  # y koordinata u pikselima
    
    return keypoints, True


def calculate_metrics(gt_keypoints: np.ndarray, mp_keypoints: np.ndarray) -> Dict[str, float]:
    """
    Izračunava metrike usporedbe između ground truth i MediaPipe keypoints.
    
    Args:
        gt_keypoints: Ground truth keypoints (13, 2)
        mp_keypoints: MediaPipe keypoints nakon mapiranja (13, 2)
    
    Returns:
        Dictionary s metrikama
    """
    # Euclidean distance za svaku točku
    distances = np.sqrt(np.sum((gt_keypoints - mp_keypoints) ** 2, axis=1))
    
    # Metrike
    metrics = {
        'mean_error': np.mean(distances),
        'median_error': np.median(distances),
        'std_error': np.std(distances),
        'rmse': np.sqrt(np.mean(distances ** 2)),
        'max_error': np.max(distances),
        'min_error': np.min(distances)
    }
    
    return metrics, distances


def visualize_comparison(image_path: str, gt_keypoints: np.ndarray, 
                        mp_keypoints_full: np.ndarray, output_path: str):
    """
    Vizualizira usporedbu ground truth i MediaPipe keypoints.
    
    Args:
        image_path: Putanja do originalne slike
        gt_keypoints: Ground truth keypoints (13, 2)
        mp_keypoints_full: Svi MediaPipe keypoints (33, 2)
        output_path: Putanja za spremanje rezultata
    """
    image = cv2.imread(image_path)
    
    # Crtanje ground truth keypoints (plavo)
    for i, (x, y) in enumerate(gt_keypoints):
        cv2.circle(image, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.putText(image, str(i), (int(x)+5, int(y)-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    # Crtanje MediaPipe keypoints za usporedbu (crveno)
    for penn_idx, mp_idx in PENNACTION_TO_MEDIAPIPE.items():
        x, y = mp_keypoints_full[mp_idx]
        cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
    
    # Dodavanje legende
    cv2.putText(image, "Plavo: Ground Truth", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(image, "Crveno: MediaPipe", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(output_path, image)


def process_single_image(image_path: str, mat_path: str, frame_idx: int = 0) -> Dict:
    """
    Procesira jednu sliku - učitava GT, detektira MediaPipe i uspoređuje.
    
    Args:
        image_path: Putanja do slike
        mat_path: Putanja do .mat datoteke
        frame_idx: Indeks framea u videu
    
    Returns:
        Dictionary s rezultatima
    """
    print(f"\nProcesuiranje: {image_path}")
    
    # 1. Učitaj ground truth
    gt_keypoints = load_pennaction_ground_truth(mat_path, frame_idx)
    print(f"  Ground truth učitan: {gt_keypoints.shape}")
    
    # 2. Detektiraj MediaPipe keypoints
    mp_keypoints_full, success = detect_mediapipe_keypoints(image_path)
    
    if not success:
        print(f"  ⚠️ MediaPipe nije uspio detektirati pozu!")
        return None
    
    print(f"  MediaPipe detektirao: {mp_keypoints_full.shape}")
    
    # 3. Mapiraj MediaPipe keypoints na PennAction format
    mp_keypoints_mapped = np.zeros((13, 2))
    for penn_idx, mp_idx in PENNACTION_TO_MEDIAPIPE.items():
        mp_keypoints_mapped[penn_idx] = mp_keypoints_full[mp_idx]
    
    # 4. Izračunaj metrike
    metrics, distances = calculate_metrics(gt_keypoints, mp_keypoints_mapped)
    
    print(f"  RMSE: {metrics['rmse']:.2f} px")
    print(f"  Mean error: {metrics['mean_error']:.2f} px")
    
    # 5. Pripremi rezultate za tablicu
    results = {
        'image': os.path.basename(image_path),
        'rmse': metrics['rmse'],
        'mean_error': metrics['mean_error'],
        'median_error': metrics['median_error'],
        'std_error': metrics['std_error'],
        'max_error': metrics['max_error'],
        'min_error': metrics['min_error'],
        'gt_keypoints': gt_keypoints,
        'mp_keypoints': mp_keypoints_mapped,
        'mp_keypoints_full': mp_keypoints_full,
        'per_keypoint_errors': distances
    }
    
    return results


def main():
    """
    Glavna funkcija koja procesira 5 slika i generira rezultate.
    """
    print("="*60)
    print("ZADATAK 1: Usporedba Ground Truth i MediaPipe Keypoints")
    print("="*60)
    
    # Definiranje putanja
    base_dir = Path("../data/raw/PennAction")
    output_dir = Path("../data/results/zadatak1")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Odabir 5 slika (primjer - prilagoditi stvarnim podacima)
    # Format: (video_id, frame_idx)
    selected_images = [
        ("0001", 50),   # Baseball pitch
        ("0100", 30),   # Tennis serve
        ("0200", 40),   # Golf swing
        ("0300", 20),   # Jumping jacks
        ("0400", 60)    # Squat
    ]
    
    all_results = []
    
    for video_id, frame_idx in selected_images:
        # Konstrukcija putanja
        image_path = base_dir / "frames" / video_id / f"{frame_idx:06d}.jpg"
        mat_path = base_dir / "labels" / f"{video_id}.mat"
        
        if not image_path.exists():
            print(f"⚠️ Slika ne postoji: {image_path}")
            continue
        
        if not mat_path.exists():
            print(f"⚠️ .mat datoteka ne postoji: {mat_path}")
            continue
        
        # Procesiranje
        result = process_single_image(str(image_path), str(mat_path), frame_idx)
        
        if result is not None:
            all_results.append(result)
            
            # Vizualizacija
            output_path = output_dir / f"comparison_{video_id}_frame{frame_idx}.jpg"
            visualize_comparison(
                str(image_path),
                result['gt_keypoints'],
                result['mp_keypoints_full'],
                str(output_path)
            )
            print(f"  ✓ Vizualizacija spremljena: {output_path}")
    
    # Generiranje sveobuhvatne tablice rezultata
    if all_results:
        print("\n" + "="*60)
        print("REZULTATI USPOREDBE")
        print("="*60)
        
        # Tablica s osnovnim metrikama
        summary_data = []
        for r in all_results:
            summary_data.append({
                'Slika': r['image'],
                'RMSE (px)': f"{r['rmse']:.2f}",
                'Mean Error (px)': f"{r['mean_error']:.2f}",
                'Median Error (px)': f"{r['median_error']:.2f}",
                'Std Error (px)': f"{r['std_error']:.2f}",
                'Max Error (px)': f"{r['max_error']:.2f}",
                'Min Error (px)': f"{r['min_error']:.2f}"
            })
        
        df_summary = pd.DataFrame(summary_data)
        print("\nOsnovne metrike po slici:")
        print(df_summary.to_string(index=False))
        
        # Spremanje u CSV
        csv_path = output_dir / "summary_results.csv"
        df_summary.to_csv(csv_path, index=False)
        print(f"\n✓ Rezultati spremljeni u: {csv_path}")
        
        # Detaljne greške po ključnim točkama
        print("\n" + "-"*60)
        print("Prosječne greške po ključnim točkama (svih 5 slika):")
        print("-"*60)
        
        # Agregiranje grešaka po keypoints
        all_errors = np.array([r['per_keypoint_errors'] for r in all_results])
        mean_errors_per_keypoint = np.mean(all_errors, axis=0)
        
        for i, (keypoint_name, error) in enumerate(zip(PENNACTION_KEYPOINT_NAMES, mean_errors_per_keypoint)):
            print(f"{i:2d}. {keypoint_name:20s}: {error:6.2f} px")
        
        # Ukupna statistika
        print("\n" + "="*60)
        print("UKUPNA STATISTIKA (svih 5 slika)")
        print("="*60)
        overall_rmse = np.mean([r['rmse'] for r in all_results])
        overall_mean = np.mean([r['mean_error'] for r in all_results])
        print(f"Prosječni RMSE:       {overall_rmse:.2f} px")
        print(f"Prosječna Mean Error: {overall_mean:.2f} px")
        print("="*60)
    
    else:
        print("\n⚠️ Niti jedna slika nije uspješno procesirana!")


if __name__ == "__main__":
    main()
