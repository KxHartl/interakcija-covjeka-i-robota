#!/usr/bin/env python3
"""
Pomoćne funkcije za mapiranje i vizualizaciju ključnih točaka.

Autor: Krešimir Hartl
Datum: Prosinac 2025
"""

import numpy as np
import cv2
from typing import List, Tuple

# PennAction keypoint definicije
PENNACTION_KEYPOINTS = {
    0: "head",
    1: "left_shoulder",
    2: "right_shoulder",
    3: "left_elbow",
    4: "right_elbow",
    5: "left_wrist",
    6: "right_wrist",
    7: "left_hip",
    8: "right_hip",
    9: "left_knee",
    10: "right_knee",
    11: "left_ankle",
    12: "right_ankle"
}

# MediaPipe Pose keypoint definicije (33 točke)
MEDIAPIPE_KEYPOINTS = {
    0: "nose",
    1: "left_eye_inner",
    2: "left_eye",
    3: "left_eye_outer",
    4: "right_eye_inner",
    5: "right_eye",
    6: "right_eye_outer",
    7: "left_ear",
    8: "right_ear",
    9: "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index"
}

# Veze između točaka za crtanje skeleta (PennAction)
PENNACTION_SKELETON = [
    (1, 3), (3, 5),  # Lijeva ruka
    (2, 4), (4, 6),  # Desna ruka
    (1, 7), (2, 8),  # Trup
    (7, 9), (9, 11), # Lijeva noga
    (8, 10), (10, 12), # Desna noga
    (1, 2), (7, 8),  # Horizontalne veze
    (0, 1), (0, 2)   # Glava
]


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray, 
                  color: Tuple[int, int, int] = (0, 255, 0),
                  radius: int = 5, thickness: int = -1) -> np.ndarray:
    """
    Crta ključne točke na slici.
    
    Args:
        image: Ulazna slika
        keypoints: Array shape (N, 2) s (x, y) koordinatama
        color: BGR boja za točke
        radius: Radijus točke
        thickness: Debljina (-1 za popunjeno)
    
    Returns:
        Slika s nacrtanim točkama
    """
    img_copy = image.copy()
    
    for i, (x, y) in enumerate(keypoints):
        if not np.isnan(x) and not np.isnan(y):
            cv2.circle(img_copy, (int(x), int(y)), radius, color, thickness)
    
    return img_copy


def draw_skeleton(image: np.ndarray, keypoints: np.ndarray,
                 connections: List[Tuple[int, int]],
                 color: Tuple[int, int, int] = (255, 255, 0),
                 thickness: int = 2) -> np.ndarray:
    """
    Crta skeletne veze između ključnih točaka.
    
    Args:
        image: Ulazna slika
        keypoints: Array shape (N, 2) s (x, y) koordinatama
        connections: Lista tupleova (idx1, idx2) koji definiraju veze
        color: BGR boja linija
        thickness: Debljina linija
    
    Returns:
        Slika s nacrtanim skeletom
    """
    img_copy = image.copy()
    
    for (idx1, idx2) in connections:
        if idx1 < len(keypoints) and idx2 < len(keypoints):
            pt1 = keypoints[idx1]
            pt2 = keypoints[idx2]
            
            # Provjera valjanosti točaka
            if not (np.isnan(pt1).any() or np.isnan(pt2).any()):
                cv2.line(img_copy,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1])),
                        color, thickness)
    
    return img_copy


def draw_keypoint_labels(image: np.ndarray, keypoints: np.ndarray,
                        labels: dict,
                        color: Tuple[int, int, int] = (255, 255, 255),
                        font_scale: float = 0.4,
                        thickness: int = 1) -> np.ndarray:
    """
    Dodaje tekstualne labele uz ključne točke.
    
    Args:
        image: Ulazna slika
        keypoints: Array shape (N, 2) s (x, y) koordinatama
        labels: Dictionary {index: "label_name"}
        color: BGR boja teksta
        font_scale: Veličina fonta
        thickness: Debljina teksta
    
    Returns:
        Slika s labelama
    """
    img_copy = image.copy()
    
    for idx, (x, y) in enumerate(keypoints):
        if idx in labels and not np.isnan(x) and not np.isnan(y):
            label = labels[idx]
            cv2.putText(img_copy, label,
                       (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, color, thickness)
    
    return img_copy


def create_side_by_side_comparison(image: np.ndarray,
                                   gt_keypoints: np.ndarray,
                                   mp_keypoints: np.ndarray,
                                   title_left: str = "Ground Truth",
                                   title_right: str = "MediaPipe") -> np.ndarray:
    """
    Kreira side-by-side usporedbu ground truth i MediaPipe keypoints.
    
    Args:
        image: Originalna slika
        gt_keypoints: Ground truth keypoints (13, 2)
        mp_keypoints: MediaPipe keypoints (13, 2) - već mapirano
        title_left: Naslov lijeve slike
        title_right: Naslov desne slike
    
    Returns:
        Kombinirana slika side-by-side
    """
    # Kreiranje dvije kopije slike
    img_left = image.copy()
    img_right = image.copy()
    
    # Crtanje ground truth (lijevo)
    img_left = draw_skeleton(img_left, gt_keypoints, PENNACTION_SKELETON, 
                            color=(0, 255, 0), thickness=2)
    img_left = draw_keypoints(img_left, gt_keypoints, color=(0, 255, 0), radius=5)
    
    # Crtanje MediaPipe (desno)
    img_right = draw_skeleton(img_right, mp_keypoints, PENNACTION_SKELETON,
                             color=(0, 0, 255), thickness=2)
    img_right = draw_keypoints(img_right, mp_keypoints, color=(0, 0, 255), radius=5)
    
    # Dodavanje naslova
    cv2.putText(img_left, title_left, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    cv2.putText(img_right, title_right, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    
    # Spajanje slika horizontalno
    combined = np.hstack([img_left, img_right])
    
    return combined


def calculate_per_keypoint_errors(gt_keypoints: np.ndarray,
                                  mp_keypoints: np.ndarray) -> dict:
    """
    Izračunava greške za svaku pojedinačnu ključnu točku.
    
    Args:
        gt_keypoints: Ground truth (13, 2)
        mp_keypoints: MediaPipe (13, 2)
    
    Returns:
        Dictionary s greškama po točkama
    """
    distances = np.sqrt(np.sum((gt_keypoints - mp_keypoints) ** 2, axis=1))
    
    errors = {}
    for idx, name in PENNACTION_KEYPOINTS.items():
        errors[name] = {
            'index': idx,
            'error_px': distances[idx],
            'gt_position': tuple(gt_keypoints[idx]),
            'mp_position': tuple(mp_keypoints[idx])
        }
    
    return errors


def print_keypoint_errors_table(errors: dict):
    """
    Ispisuje tablicu grešaka po ključnim točkama.
    
    Args:
        errors: Dictionary iz calculate_per_keypoint_errors()
    """
    print("\n" + "="*80)
    print(f"{'Keypoint':<20} | {'Error (px)':>12} | {'GT Position':>20} | {'MP Position':>20}")
    print("="*80)
    
    for name, data in sorted(errors.items(), key=lambda x: x[1]['error_px'], reverse=True):
        gt_pos = f"({data['gt_position'][0]:.1f}, {data['gt_position'][1]:.1f})"
        mp_pos = f"({data['mp_position'][0]:.1f}, {data['mp_position'][1]:.1f})"
        print(f"{name:<20} | {data['error_px']:>12.2f} | {gt_pos:>20} | {mp_pos:>20}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Testna vizualizacija
    print("PennAction keypoints:")
    for idx, name in PENNACTION_KEYPOINTS.items():
        print(f"  {idx:2d}: {name}")
    
    print("\nMediaPipe keypoints (relevantni za mapiranje):")
    relevant_mp = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
    for idx in relevant_mp:
        print(f"  {idx:2d}: {MEDIAPIPE_KEYPOINTS[idx]}")
