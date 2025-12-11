#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zadatak 1: Vizualizacija - Usporedba Ground Truth i MediaPipe predikcija

Vizualizira 3D koordinatne osi, MediaPipe keypointe i predikcije na slikama.
Dataset: HPE-24 sa frameima 50, 150, 250, 350, 420, 508

Autor: Student
Datum: 2025
"""

import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
from typing import Dict, Tuple, List

# ============================================================================
# KONSTANTE
# ============================================================================

K = np.array([
    [575.816, 0.0, 320.0],
    [0.0, 575.816, 240.0],
    [0.0, 0.0, 1.0]
], dtype=np.float32)

# Dostupni framea u HPE-24 datasetu
AVAILABLE_FRAMES = [50, 150, 250, 350, 420, 508]

# Boje za vizualizaciju
COLOR_RED = (0, 0, 255)      # BGR: crvena
COLOR_GREEN = (0, 255, 0)    # BGR: zelena
COLOR_BLUE = (255, 0, 0)     # BGR: plava
COLOR_YELLOW = (0, 255, 255) # BGR: Žuta
COLOR_CYAN = (255, 255, 0)   # BGR: cyan
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

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

    return R.astype(np.float32), T_mm.astype(np.float32) / 1000.0  # u metre


def project_point(P: np.ndarray, K_mat: np.ndarray = K) -> Tuple[int, int]:
    """Projektira 3D točku u 2D image koordinate."""
    p = K_mat @ P
    u = p[0] / p[2]
    v = p[1] / p[2]
    return int(round(u)), int(round(v))


def is_point_in_image(u: int, v: int, img_shape: Tuple[int, int]) -> bool:
    """Provjerava je li točka u image granicama."""
    h, w = img_shape[:2]
    return 0 <= u < w and 0 <= v < h


# ============================================================================
# VIZUALIZACIJSKE FUNKCIJE
# ============================================================================

class VisualizerTask1:
    """Klasa za vizualizaciju Zadatka 1."""

    def __init__(self, K_mat: np.ndarray = K):
        self.K = K_mat
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6
        )

    def __del__(self):
        """Očisti resurse."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

    def draw_axes_3d(self, img: np.ndarray, R: np.ndarray, T_m: np.ndarray,
                     axis_len_m: float = 0.2, thickness: int = 2) -> np.ndarray:
        """Crta 3D koordinatne osi na slici (Ground Truth)."""
        img_copy = img.copy()
        T_mm = T_m * 1000.0
        axis_len_mm = axis_len_m * 1000.0

        # Ishodište
        P0 = T_mm

        # Lokalne osi
        x_axis = R[:, 0]
        y_axis = R[:, 1]
        z_axis = R[:, 2]

        # Krajnje točke
        PX = T_mm + axis_len_mm * x_axis
        PY = T_mm + axis_len_mm * y_axis
        PZ = T_mm + axis_len_mm * z_axis

        # Projektiraj
        u0, v0 = project_point(P0, self.K)
        ux, vx = project_point(PX, self.K)
        uy, vy = project_point(PY, self.K)
        uz, vz = project_point(PZ, self.K)

        # Nacrtaj
        if is_point_in_image(u0, v0, img_copy.shape):
            cv2.circle(img_copy, (u0, v0), 5, COLOR_YELLOW, -1)
            
            if is_point_in_image(ux, vx, img_copy.shape):
                cv2.line(img_copy, (u0, v0), (ux, vx), COLOR_RED, thickness)  # X
                cv2.putText(img_copy, 'X', (ux+5, vx+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RED, 2)
            
            if is_point_in_image(uy, vy, img_copy.shape):
                cv2.line(img_copy, (u0, v0), (uy, vy), COLOR_GREEN, thickness)  # Y
                cv2.putText(img_copy, 'Y', (uy+5, vy+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GREEN, 2)
            
            if is_point_in_image(uz, vz, img_copy.shape):
                cv2.line(img_copy, (u0, v0), (uz, vz), COLOR_BLUE, thickness)  # Z
                cv2.putText(img_copy, 'Z', (uz+5, vz+5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLUE, 2)

        return img_copy

    def draw_mediapipe_landmarks(self, img: np.ndarray, 
                                 landmarks_2d: np.ndarray,
                                 radius: int = 2,
                                 color: Tuple[int, int, int] = COLOR_CYAN) -> np.ndarray:
        """Crta MediaPipe keypointe na slici."""
        img_copy = img.copy()
        h, w = img.shape[:2]

        for i, (x, y) in enumerate(landmarks_2d):
            x, y = int(x), int(y)
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(img_copy, (x, y), radius, color, -1)

        return img_copy

    def get_face_bounding_box(self, landmarks_2d: np.ndarray) -> Dict:
        """Izračunava bounding box oko lica."""
        if len(landmarks_2d) == 0:
            return {'found': False}

        x_coords = landmarks_2d[:, 0]
        y_coords = landmarks_2d[:, 1]

        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())

        return {
            'found': True,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'width': x_max - x_min,
            'height': y_max - y_min
        }

    def draw_face_bbox(self, img: np.ndarray, bbox: Dict,
                      color: Tuple[int, int, int] = COLOR_GREEN,
                      thickness: int = 2) -> np.ndarray:
        """Crta bounding box oko lica."""
        if not bbox.get('found', False):
            return img

        img_copy = img.copy()
        pt1 = (bbox['x_min'], bbox['y_min'])
        pt2 = (bbox['x_max'], bbox['y_max'])

        cv2.rectangle(img_copy, pt1, pt2, color, thickness)

        return img_copy

    def extract_mediapipe_landmarks(self, image_rgb: np.ndarray) -> Dict:
        """Ekstraktuje MediaPipe landmarks iz slike."""
        results = self.face_mesh.process(image_rgb)

        result_dict = {
            'detected': False,
            'landmarks_2d': [],
            'landmarks_3d': [],
            'count': 0
        }

        if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
            face = results.multi_face_landmarks[0]
            h, w = image_rgb.shape[:2]

            landmarks_2d = []
            landmarks_3d = []

            for lm in face.landmark:
                x = lm.x * w
                y = lm.y * h
                z = lm.z
                landmarks_2d.append([x, y])
                landmarks_3d.append([x, y, z])

            result_dict['detected'] = True
            result_dict['landmarks_2d'] = np.array(landmarks_2d)
            result_dict['landmarks_3d'] = np.array(landmarks_3d)
            result_dict['count'] = len(face.landmark)

        return result_dict

    def draw_comparison_image(self, img: np.ndarray, 
                             R_gt: np.ndarray = None,
                             T_gt: np.ndarray = None,
                             mp_landmarks: np.ndarray = None,
                             show_bbox: bool = True) -> np.ndarray:
        """Crta kompletnu usporedbu na jednoj slici."""
        img_vis = img.copy()

        # Crta Ground Truth osi
        if R_gt is not None and T_gt is not None:
            img_vis = self.draw_axes_3d(img_vis, R_gt, T_gt, axis_len_m=0.15)

        # Crta MediaPipe keypointe
        if mp_landmarks is not None and len(mp_landmarks) > 0:
            img_vis = self.draw_mediapipe_landmarks(img_vis, mp_landmarks, 
                                                   radius=1, color=COLOR_CYAN)
            
            # Crta bounding box
            if show_bbox:
                bbox = self.get_face_bounding_box(mp_landmarks)
                img_vis = self.draw_face_bbox(img_vis, bbox, color=COLOR_GREEN, thickness=1)

        return img_vis

    def add_text_info(self, img: np.ndarray, frame_id: int,
                     has_gt: bool, has_mp: bool) -> np.ndarray:
        """Dodaje tekst informacije na sliku."""
        img_copy = img.copy()
        h, w = img.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color_text = COLOR_WHITE
        bg_color = COLOR_BLACK

        # Frame ID
        text = f"Frame: {frame_id:05d}"
        cv2.putText(img_copy, text, (10, 25), font, font_scale, color_text, thickness)

        # Status
        status = f"GT: {'DA' if has_gt else 'NE':3} | MP: {'DA' if has_mp else 'NE':3}"
        cv2.putText(img_copy, status, (10, 45), font, font_scale, color_text, thickness)

        # Legenda
        y_offset = h - 60
        cv2.putText(img_copy, "Legenda:", (10, y_offset), font, font_scale, color_text, thickness)
        
        # Red - X os
        cv2.line(img_copy, (10, y_offset + 20), (30, y_offset + 20), COLOR_RED, 2)
        cv2.putText(img_copy, "X (Ground Truth)", (35, y_offset + 25), font, 0.4, COLOR_RED, 1)
        
        # Cyan - MediaPipe
        cv2.circle(img_copy, (15, y_offset + 40), 2, COLOR_CYAN, -1)
        cv2.putText(img_copy, "MediaPipe keypointi", (35, y_offset + 45), font, 0.4, COLOR_CYAN, 1)

        return img_copy


# ============================================================================
# DEMO / MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("TASK 1 - Vizualizacija usporedbe Ground Truth i MediaPipe")
    print("="*70 + "\n")

    # Koristi data folder HPE-24
    data_root = Path("../data/raw/hpe/24")

    if not data_root.exists():
        print(f"⚠️  Putanja ne postoji: {data_root}")
        print("Prilagodi putanju prema svojoj strukturi direktorija.\n")
        exit(1)

    # Kreiraj visualizer
    visualizer = VisualizerTask1(K)

    # Procesiraj sve dostupne framee
    print(f"Dostupni framea: {AVAILABLE_FRAMES}\n")
    
    for frame_id in AVAILABLE_FRAMES:
        rgb_path = data_root / f"frame_{frame_id:05d}_rgb.png"
        pose_path = data_root / f"frame_{frame_id:05d}_pose.txt"

        if not rgb_path.exists():
            print(f"[Frame {frame_id:05d}] RGB datoteka ne postoji\n")
            continue

        # Učitaj sliku
        img_bgr = cv2.imread(str(rgb_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        print(f"Učitana slika: {rgb_path.name}")
        print(f"Dimenzije: {img_bgr.shape[1]} x {img_bgr.shape[0]}")

        # Ground Truth
        R_gt, T_gt = None, None
        has_gt = False
        if pose_path.exists():
            try:
                R_gt, T_gt = load_biwi_pose_txt(str(pose_path))
                has_gt = True
                print(f"Ground Truth pose učitan")
            except Exception as e:
                print(f"Greška pri učitavanju pose: {e}")

        # MediaPipe
        mp_result = visualizer.extract_mediapipe_landmarks(img_rgb)
        print(f"MediaPipe: {'Detektiran' if mp_result['detected'] else 'Nije detektiran'}")
        if mp_result['detected']:
            print(f"  - Broj keypointa: {mp_result['count']}")

        # Vizualizacija
        img_vis = visualizer.draw_comparison_image(
            img_bgr,
            R_gt=R_gt,
            T_gt=T_gt,
            mp_landmarks=mp_result['landmarks_2d'] if mp_result['detected'] else None
        )

        # Dodaj tekst
        img_vis = visualizer.add_text_info(img_vis, frame_id, has_gt, mp_result['detected'])

        # Spremi vizualizaciju
        output_path = Path(f"./results/task_1_visualization_frame_{frame_id:05d}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), img_vis)
        print(f"✓ Vizualizacija spremljena u: {output_path.name}\n")

    print("="*70)
    print("✓ Sve vizualizacije su generirane!\n")
