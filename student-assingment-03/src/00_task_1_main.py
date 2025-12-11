#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zadatak 1: Usporedba Ground Truth podataka i MediaPipe predikcija

Ovaj skript učitava ground truth podatke o poziciji i rotaciji glave,
uspoređuje ih sa MediaPipe predikcijama i analizira točnost detektiranja.

Dataset: HPE-24 sa frameima 50, 150, 250, 350, 420, 508

Autor: Student
Datum: 2025
"""

import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Tuple, Dict, List
import json
from datetime import datetime

# ============================================================================
# KONSTANTE I KALIBRACIJA
# ============================================================================

# Kalibracija BIWI kamere (procjena)
K = np.array([
    [575.816, 0.0, 320.0],
    [0.0, 575.816, 240.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Dostupni framea u HPE-24 datasetu
AVAILABLE_FRAMES = [50, 150, 250, 350, 420, 508]

# ============================================================================
# POMOČNE FUNKCIJE
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
    import math

    assert R.shape == (3, 3)

    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])   # oko X
        yaw   = math.atan2(-R[2, 0], sy)       # oko Y
        roll  = math.atan2(R[1, 0], R[0, 0])   # oko Z
    else:
        # gimbal lock slučaj (rijetko)
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0

    return np.degrees(np.array([pitch, yaw, roll]))


def project_point(P: np.ndarray, K_mat: np.ndarray = K) -> Tuple[int, int]:
    """Projektira 3D točku u 2D image koordinate."""
    p = K_mat @ P
    u = p[0] / p[2]
    v = p[1] / p[2]
    return int(round(u)), int(round(v))


def extract_mediapipe_head_pose(image_rgb: np.ndarray) -> Dict:
    """Ekstraktuje head pose iz MediaPipe FaceMesh detektora."""
    mp_face_mesh = mp.solutions.face_mesh
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6
    )

    results = face_mesh.process(image_rgb)
    face_mesh.close()

    result_dict = {
        'detected': False,
        'landmarks': None,
        'landmark_count': 0,
        'head_box': None
    }

    if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
        face = results.multi_face_landmarks[0]
        result_dict['detected'] = True
        result_dict['landmark_count'] = len(face.landmark)
        
        # Ekstraktiraj koordinate
        landmarks = []
        h, w = image_rgb.shape[:2]
        for lm in face.landmark:
            x = lm.x * w
            y = lm.y * h
            z = lm.z
            landmarks.append([x, y, z])
        
        result_dict['landmarks'] = np.array(landmarks)

    return result_dict


# ============================================================================
# ANALIZA I USPOREDBA
# ============================================================================

class Task1Analyzer:
    """Klasa za analizu Zadatka 1: usporedba ground truth i MediaPipe."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.results = []

    def process_frame(self, frame_id: int) -> Dict:
        """Procesira jedan frame: učitava ground truth i MediaPipe predikcije."""
        
        result = {
            'frame_id': frame_id,
            'status': 'success',
            'error': None,
            'ground_truth': {},
            'mediapipe': {},
            'comparison': {}
        }

        try:
            # Učitaj slike
            rgb_path = self.data_root / f"frame_{frame_id:05d}_rgb.png"
            if not rgb_path.exists():
                result['status'] = 'file_not_found'
                result['error'] = f"RGB datoteka ne postoji: {rgb_path.name}"
                return result

            image_bgr = cv2.imread(str(rgb_path))
            if image_bgr is None:
                result['status'] = 'failed_to_load'
                result['error'] = f"Ne mogu učitati sliku: {rgb_path.name}"
                return result

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w = image_bgr.shape[:2]

            # Učitaj ground truth pozu
            pose_path = self.data_root / f"frame_{frame_id:05d}_pose.txt"
            if pose_path.exists():
                try:
                    R_gt, T_gt = load_biwi_pose_txt(str(pose_path))
                    euler_gt = rotmat_to_euler_xyz(R_gt)
                    
                    result['ground_truth'] = {
                        'rotation_matrix': R_gt.tolist(),
                        'translation_m': T_gt.tolist(),
                        'euler_angles_deg': euler_gt.tolist()
                    }
                except Exception as e:
                    result['ground_truth'] = {'error': str(e)}
            else:
                result['ground_truth'] = {'available': False}

            # Ekstraktiraj MediaPipe head pose
            mp_result = extract_mediapipe_head_pose(image_rgb)
            result['mediapipe'] = mp_result

            # Usporedba
            if result['ground_truth'] and 'error' not in result['ground_truth'] and mp_result['detected']:
                result['comparison'] = {
                    'ground_truth_available': True,
                    'mediapipe_detected': True,
                    'ground_truth_euler': result['ground_truth'].get('euler_angles_deg'),
                    'mediapipe_landmarks_count': mp_result['landmark_count']
                }
            else:
                result['comparison'] = {
                    'ground_truth_available': 'euler_angles_deg' in result.get('ground_truth', {}),
                    'mediapipe_detected': mp_result['detected']
                }

        except Exception as e:
            result['status'] = 'error'
            result['error'] = str(e)

        return result

    def process_frames_list(self, frame_list: List[int]) -> None:
        """Procesira listu framea."""
        print(f"\n{'='*70}")
        print(f"Analiza Zadatka 1: Ground Truth vs MediaPipe")
        print(f"{'='*70}")
        print(f"\nData folder: {self.data_root}")
        print(f"Dostupni framea: {sorted(frame_list)}")
        print(f"Processiranje {len(frame_list)} framea...\n")

        for frame_id in sorted(frame_list):
            result = self.process_frame(frame_id)
            self.results.append(result)

            status_str = f"[Frame {frame_id:05d}] Status: {result['status']}"
            if result['status'] == 'success':
                gt_available = 'euler_angles_deg' in result.get('ground_truth', {})
                mp_detected = result['mediapipe'].get('detected', False)
                status_str += f" | GT: {gt_available} | MP: {mp_detected}"
            
            print(status_str)
            if result['error']:
                print(f"  └─ Greška: {result['error']}")

    def save_results(self, output_path: Path) -> None:
        """Sprema rezultate u JSON datoteku."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Pripremi podatke za JSON (konvertiraj numpy nizove)
        json_results = []
        for r in self.results:
            json_r = r.copy()
            json_results.append(json_r)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'total_frames': len(self.results),
                'data_folder': str(self.data_root),
                'available_frames': AVAILABLE_FRAMES,
                'results': json_results
            }, f, indent=2, ensure_ascii=False)

        print(f"\nRezultati spravljeni u: {output_path}")

    def print_summary(self) -> None:
        """Ispisuje sažetak rezultata."""
        print(f"\n{'='*70}")
        print("SAŽETAK ANALIZE")
        print(f"{'='*70}")

        total = len(self.results)
        successful = sum(1 for r in self.results if r['status'] == 'success')
        gt_available = sum(1 for r in self.results 
                          if 'euler_angles_deg' in r.get('ground_truth', {}))
        mp_detected = sum(1 for r in self.results 
                         if r['mediapipe'].get('detected', False))

        print(f"\nUkupno framea: {total}")
        print(f"Uspješno procesiranih: {successful}")
        print(f"Ground truth dostupnih: {gt_available}")
        print(f"MediaPipe detektiranih: {mp_detected}")

        if gt_available > 0 and mp_detected > 0:
            print(f"\nFramea sa oba podatka: ", end="")
            both = sum(1 for r in self.results
                      if 'euler_angles_deg' in r.get('ground_truth', {})
                      and r['mediapipe'].get('detected', False))
            print(f"{both}")

        print(f"\n{'='*70}\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Koristi data folder HPE-24
    data_root = Path("./student-assingment-03/data/raw/hpe/24")
    
    # Ako putanja ne postoji, ispiši upozorenje
    if not data_root.exists():
        print(f"\n⚠️  Putanja ne postoji: {data_root}")
        print("\nPrilagodi 'data_root' putanju prema svojoj strukturi direktorija.")
        print("Primjer:\n  data_root = Path('./student-assingment-03/data/raw/hpe/24')")
        print("\nZavršavam...\n")
        exit(1)

    # Kreiraj analizator
    analyzer = Task1Analyzer(data_root)

    # Procesiraj dostupne framee iz HPE-24 dataseta
    # Dostupni framea: 50, 150, 250, 350, 420, 508
    analyzer.process_frames_list(AVAILABLE_FRAMES)
    analyzer.print_summary()

    # Spremi rezultate
    output_json = Path("./results/task_1_analysis.json")
    analyzer.save_results(output_json)

    print("✓ Analiza Zadatka 1 je završena!")
