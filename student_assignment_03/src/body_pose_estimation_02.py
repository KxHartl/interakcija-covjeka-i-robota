"""
Samostalni zadatak 2: Izračun kuta zgloba
- Odabrati 5 fotografija iz mape podaci
- Izračunati kutove:
  * desni lakat
  * lijevo koljeno
- Ispisati rezultate u terminal i prikazati tablicu
- Rezultate spremiti u .csv obliku u outputs
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from pathlib import Path

# Konfiguracija
DATA_DIR = 'data/raw/1137'
IMAGE_INDICES = [1, 5, 10, 15, 20]  # 5 fotografija
OUTPUT_FILE = 'outputs/body_pose_estimation_02_results.csv'


def calculate_angle(point_a, point_b, point_c):
    """
    Izračunaj kut između tri točke koristeći vektorsku matematiku.
    
    Kut se računa kao: ∠ABC gdje je B vertex (srednja točka)
    
    Args:
        point_a: tuple (x, y) - prva točka
        point_b: tuple (x, y) - vertex (srednja točka)
        point_c: tuple (x, y) - treća točka
    
    Returns:
        float: kut u stupnjima (0-180°)
    """
    # Pretvori u numpy arrays
    a = np.array(point_a)
    b = np.array(point_b)
    c = np.array(point_c)
    
    # Kreiraj vektore
    ba = a - b  # Vektor od B do A
    bc = c - b  # Vektor od B do C
    
    # Izračunaj kut koristeći dot product
    # cos(θ) = (BA · BC) / (|BA| * |BC|)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Ograniči na [-1, 1] radi numeričke stabilnosti
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Konvertuj u radijane i zatim u stupnje
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg


def get_pose_landmarks(image_path):
    """
    Detektuj pose landmarke na slici koristeći MediaPipe.
    
    Args:
        image_path: putanja do slike
    
    Returns:
        dict: rječnik landmarkova ili None ako nije detektirana poza
    """
    slika = cv2.imread(image_path)
    
    if slika is None:
        print(f"Ne mogu učitati sliku: {image_path}")
        return None
    
    h, w = slika.shape[:2]
    
    # MediaPipe Pose inicijalizacija
    mp_pose = mp.solutions.pose
    
    with mp_pose.Pose(static_image_mode=True) as pose:
        slika_rgb = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
        results = pose.process(slika_rgb)
    
    if not results.pose_landmarks:
        print(f"Nije detektirana poza na slici: {image_path}")
        return None
    
    # Pretvori u piksele i spremi u rječnik
    landmarks = {}
    for i, landmark in enumerate(results.pose_landmarks.landmark):
        x_pixel = int(landmark.x * w)
        y_pixel = int(landmark.y * h)
        landmarks[i] = (x_pixel, y_pixel)
    
    return landmarks


def calculate_right_elbow_angle(landmarks):
    """
    Izračunaj kut desnog lakta.
    
    Lakat je vertex (srednja točka):
    - Rameno (RIGHT_SHOULDER) - 12
    - Lakat (RIGHT_ELBOW) - 11
    - Zapis (RIGHT_WRIST) - 10
    
    Args:
        landmarks: rječnik landmarkova
    
    Returns:
        float: kut lakta u stupnjima
    """
    if None in [landmarks.get(12), landmarks.get(11), landmarks.get(10)]:
        return None
    
    shoulder = landmarks[12]  # RIGHT_SHOULDER
    elbow = landmarks[11]     # RIGHT_ELBOW
    wrist = landmarks[10]     # RIGHT_WRIST
    
    angle = calculate_angle(shoulder, elbow, wrist)
    return angle


def calculate_left_knee_angle(landmarks):
    """
    Izračunaj kut lijevog koljena.
    
    Koljeno je vertex (srednja točka):
    - Kuk (LEFT_HIP) - 23
    - Koljeno (LEFT_KNEE) - 25
    - Gležanj (LEFT_ANKLE) - 27
    
    Args:
        landmarks: rječnik landmarkova
    
    Returns:
        float: kut koljena u stupnjima
    """
    if None in [landmarks.get(23), landmarks.get(25), landmarks.get(27)]:
        return None
    
    hip = landmarks[23]      # LEFT_HIP
    knee = landmarks[25]     # LEFT_KNEE
    ankle = landmarks[27]    # LEFT_ANKLE
    
    angle = calculate_angle(hip, knee, ankle)
    return angle


def main():
    """
    Glavna funkcija za izračun kutova zglobova
    """
    
    print("=" * 80)
    print("IZRAČUN KUTOVA ZGLOBOVA")
    print("=" * 80)
    print(f"\nOdabrane fotografije: {IMAGE_INDICES}")
    print(f"Kutovi koji se računaju:")
    print(f"  - Desni lakat (shoulder-elbow-wrist)")
    print(f"  - Lijevo koljeno (hip-knee-ankle)")
    print("\n" + "=" * 80)
    
    results = []
    
    for image_idx in IMAGE_INDICES:
        # Formatiranje imena slike
        image_name = f"{image_idx:06d}.jpg"
        image_path = os.path.join(DATA_DIR, image_name)
        
        print(f"\nObrađujem sliku: {image_name}")
        
        # Provjera postoji li slika
        if not os.path.exists(image_path):
            print(f"  ⚠️  Slika ne postoji: {image_path}")
            continue
        
        # Detektuj pose landmarke
        landmarks = get_pose_landmarks(image_path)
        
        if landmarks is None:
            print(f"  ⚠️  Nije detektirana poza na slici {image_name}")
            continue
        
        # Izračunaj kutove
        right_elbow_angle = calculate_right_elbow_angle(landmarks)
        left_knee_angle = calculate_left_knee_angle(landmarks)
        
        if right_elbow_angle is None or left_knee_angle is None:
            print(f"  ⚠️  Ne mogu izračunati kutove za sliku {image_name}")
            continue
        
        # Spremi rezultat
        result = {
            'image_id': image_name,
            'right_elbow_angle': right_elbow_angle,
            'left_knee_angle': left_knee_angle
        }
        results.append(result)
        
        print(f"  ✓ Desni lakat: {right_elbow_angle:.2f}°")
        print(f"  ✓ Lijevo koljeno: {left_knee_angle:.2f}°")
    
    # Ispis tablice rezultata
    print("\n" + "=" * 80)
    print("REZULTATI")
    print("=" * 80)
    print()
    
    if results:
        # Ispis zaglavlja tablice
        print(f"{'ID Slike':<15} {'Lakat (°)':<15} {'Koljeno (°)':<15}")
        print("-" * 50)
        
        # Ispis redaka tablice
        for r in results:
            print(f"{r['image_id']:<15} {r['right_elbow_angle']:<15.2f} {r['left_knee_angle']:<15.2f}")
        
        # Ispis statistike
        print("-" * 50)
        elbow_angles = [r['right_elbow_angle'] for r in results]
        knee_angles = [r['left_knee_angle'] for r in results]
        
        print(f"\n{'STATISTIKA - DESNI LAKAT':<30}")
        print("-" * 50)
        print(f"{'Prosječni kut':<30} {np.mean(elbow_angles):.2f}°")
        print(f"{'Minimalni kut':<30} {np.min(elbow_angles):.2f}°")
        print(f"{'Maksimalni kut':<30} {np.max(elbow_angles):.2f}°")
        print(f"{'Standardna devijacija':<30} {np.std(elbow_angles):.2f}°")
        
        print(f"\n{'STATISTIKA - LIJEVO KOLJENO':<30}")
        print("-" * 50)
        print(f"{'Prosječni kut':<30} {np.mean(knee_angles):.2f}°")
        print(f"{'Minimalni kut':<30} {np.min(knee_angles):.2f}°")
        print(f"{'Maksimalni kut':<30} {np.max(knee_angles):.2f}°")
        print(f"{'Standardna devijacija':<30} {np.std(knee_angles):.2f}°")
    else:
        print("Nema rezultata za prikaz!")
    
    print("\n" + "=" * 80)
    
    # Spremi rezultate u CSV datoteku
    os.makedirs('outputs', exist_ok=True)
    
    if results:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['ID Slike', 'Lakat (°)', 'Koljeno (°)']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Ispis zaglavlja
            writer.writeheader()
            
            # Ispis redaka
            for r in results:
                writer.writerow({
                    'ID Slike': r['image_id'],
                    'Lakat (°)': f"{r['right_elbow_angle']:.2f}",
                    'Koljeno (°)': f"{r['left_knee_angle']:.2f}"
                })
        
        print(f"✓ Rezultati su spremljeni u: {OUTPUT_FILE}")
    
    print()


if __name__ == "__main__":
    main()
