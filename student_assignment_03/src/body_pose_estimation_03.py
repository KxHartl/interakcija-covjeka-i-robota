"""
Samostalni zadatak 3: Nagib tijela
- Odabrati 5 fotografija iz mape podaci
- Sami odabrati ključne točke torza
- Izračunati nagib u odnosu na vertikalu
- Rezultat prikazati u tablici: ID slike | Nagib
- Rezultate spremiti u .csv obliku u outputs

ČDИ TORZA (odabrane za analizu nagiba):
- Lijevo rameno (LEFT_SHOULDER) - index 11
- Desno rameno (RIGHT_SHOULDER) - index 12
- Lijevi kuk (LEFT_HIP) - index 23
- Desni kuk (RIGHT_HIP) - index 24

NAGIB TORZA:
Izračunava se kao kut linije od srednje točke ramena do srednje točke kukova
u odnosu na vertikalu (90°).

Ako je nagib 0° = osoba je potpuno vertikalna
Ako je nagib 45° = osoba se klanja 45° od vertikale
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
OUTPUT_FILE = 'outputs/body_pose_estimation_03_results.csv'

# Landmarki torza
# Ramena
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
# Kukovi
LEFT_HIP = 23
RIGHT_HIP = 24


def calculate_body_tilt(landmarks):
    """
    Izračunaj nagib tijela u odnosu na vertikalu.
    
    Nagib se izračunava kao kut između:
    - Linije koja spaja srednju točku ramena i srednju točku kukova
    - Vertikalne linije (90 stupnjeva)
    
    Args:
        landmarks: rječnik landmarkova (index: (x, y))
    
    Returns:
        float: nagib u stupnjima (-90 do +90°)
               0° = potpuno vertikalno
               +45° = klanja se udesno
               -45° = klanja se ulijevo
    """
    # Dohvati landmarke
    left_shoulder = landmarks.get(LEFT_SHOULDER)
    right_shoulder = landmarks.get(RIGHT_SHOULDER)
    left_hip = landmarks.get(LEFT_HIP)
    right_hip = landmarks.get(RIGHT_HIP)
    
    # Provjera da li su svi landmarki dostupni
    if None in [left_shoulder, right_shoulder, left_hip, right_hip]:
        return None
    
    # Izračunaj srednju točku ramena
    shoulder_midpoint = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    )
    
    # Izračunaj srednju točku kukova
    hip_midpoint = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    )
    
    # Kreiraj vektor od kukova prema ramenima (od dole prema gore)
    # Y-osa je invertirana u slikama (0 na vrhu, max na dnu)
    dx = shoulder_midpoint[0] - hip_midpoint[0]
    dy = shoulder_midpoint[1] - hip_midpoint[1]
    
    # Izračunaj kut u odnosu na vertikalu
    # atan2(dx, -dy) jer je vertikala u smjeru -y (prema vrhu slike)
    angle_rad = np.arctan2(dx, -dy)
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


def get_torso_visualization_info(landmarks):
    """
    Dohvati informacije za vizualizaciju torza (ključne točke).
    
    Args:
        landmarks: rječnik landmarkova
    
    Returns:
        dict: informacije o torzu
    """
    left_shoulder = landmarks.get(LEFT_SHOULDER)
    right_shoulder = landmarks.get(RIGHT_SHOULDER)
    left_hip = landmarks.get(LEFT_HIP)
    right_hip = landmarks.get(RIGHT_HIP)
    
    if None in [left_shoulder, right_shoulder, left_hip, right_hip]:
        return None
    
    shoulder_midpoint = (
        (left_shoulder[0] + right_shoulder[0]) / 2,
        (left_shoulder[1] + right_shoulder[1]) / 2
    )
    
    hip_midpoint = (
        (left_hip[0] + right_hip[0]) / 2,
        (left_hip[1] + right_hip[1]) / 2
    )
    
    return {
        'left_shoulder': left_shoulder,
        'right_shoulder': right_shoulder,
        'left_hip': left_hip,
        'right_hip': right_hip,
        'shoulder_midpoint': shoulder_midpoint,
        'hip_midpoint': hip_midpoint
    }


def main():
    """
    Glavna funkcija za izračun nagiba tijela
    """
    
    print("=" * 90)
    print("IZRAČUN NAGIBA TIJELA")
    print("=" * 90)
    print(f"\nOdabrane fotografije: {IMAGE_INDICES}")
    print(f"\nKljučne točke torza (odabrane):")
    print(f"  - Lijevo rameno (LEFT_SHOULDER) - index {LEFT_SHOULDER}")
    print(f"  - Desno rameno (RIGHT_SHOULDER) - index {RIGHT_SHOULDER}")
    print(f"  - Lijevi kuk (LEFT_HIP) - index {LEFT_HIP}")
    print(f"  - Desni kuk (RIGHT_HIP) - index {RIGHT_HIP}")
    print(f"\nNačin izračuna:")
    print(f"  1. Izračunaj srednju točku ramena")
    print(f"  2. Izračunaj srednju točku kukova")
    print(f"  3. Kreiraj vektor od kukova prema ramenima")
    print(f"  4. Izračunaj kut u odnosu na vertikalu")
    print(f"\nInterpretacija nagiba:")
    print(f"  - 0° = osoba je potpuno vertikalna")
    print(f"  - pozitivan kut = osoba se klanja udesno")
    print(f"  - negativan kut = osoba se klanja ulijevo")
    print(f"\n" + "=" * 90)
    
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
        
        # Dohvati vizualizacijske informacije
        torso_info = get_torso_visualization_info(landmarks)
        
        if torso_info is None:
            print(f"  ⚠️  Ne mogu dohvatiti landmarke torza za sliku {image_name}")
            continue
        
        # Izračunaj nagib tijela
        tilt_angle = calculate_body_tilt(landmarks)
        
        if tilt_angle is None:
            print(f"  ⚠️  Ne mogu izračunati nagib za sliku {image_name}")
            continue
        
        # Spremi rezultat
        result = {
            'image_id': image_name,
            'tilt_angle': tilt_angle,
            'torso_info': torso_info
        }
        results.append(result)
        
        # Interpretacija nagiba
        if abs(tilt_angle) < 5:
            direction = "okomito (malo varijacije)"
        elif tilt_angle > 0:
            direction = f"udesno za {abs(tilt_angle):.2f}°"
        else:
            direction = f"ulijevo za {abs(tilt_angle):.2f}°"
        
        print(f"  ✓ Nagib: {tilt_angle:+.2f}° ({direction})")
        print(f"  ✓ Srednja točka ramena: ({torso_info['shoulder_midpoint'][0]:.0f}, {torso_info['shoulder_midpoint'][1]:.0f})")
        print(f"  ✓ Srednja točka kukova: ({torso_info['hip_midpoint'][0]:.0f}, {torso_info['hip_midpoint'][1]:.0f})")
    
    # Ispis tablice rezultata
    print("\n" + "=" * 90)
    print("REZULTATI")
    print("=" * 90)
    print()
    
    if results:
        # Ispis zaglavlja tablice
        print(f"{'ID Slike':<15} {'Nagib (°)':<15} {'Interpretacija':<50}")
        print("-" * 80)
        
        # Ispis redaka tablice
        for r in results:
            angle = r['tilt_angle']
            
            if abs(angle) < 5:
                interpretation = "Okomito (malo varijacije)"
            elif angle > 0:
                interpretation = f"Nagib udesno za {abs(angle):.2f}°"
            else:
                interpretation = f"Nagib ulijevo za {abs(angle):.2f}°"
            
            print(f"{r['image_id']:<15} {angle:+<14.2f} {interpretation:<50}")
        
        # Ispis statistike
        print("-" * 80)
        angles = [r['tilt_angle'] for r in results]
        
        print(f"\n{'STATISTIKA':<30} {'VRIJEDNOST':<15}")
        print("-" * 50)
        print(f"{'Prosječni nagib':<30} {np.mean(angles):+.2f}°")
        print(f"{'Minimalni nagib':<30} {np.min(angles):+.2f}°")
        print(f"{'Maksimalni nagib':<30} {np.max(angles):+.2f}°")
        print(f"{'Standardna devijacija':<30} {np.std(angles):.2f}°")
        print(f"{'Broj obradenih slika':<30} {len(results)}")
    else:
        print("Nema rezultata za prikaz!")
    
    print("\n" + "=" * 90)
    
    # Spremi rezultate u CSV datoteku
    os.makedirs('outputs', exist_ok=True)
    
    if results:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['ID Slike', 'Nagib (°)', 'Interpretacija']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Ispis zaglavlja
            writer.writeheader()
            
            # Ispis redaka
            for r in results:
                angle = r['tilt_angle']
                
                if abs(angle) < 5:
                    interpretation = "Okomito (malo varijacije)"
                elif angle > 0:
                    interpretation = f"Nagib udesno za {abs(angle):.2f}°"
                else:
                    interpretation = f"Nagib ulijevo za {abs(angle):.2f}°"
                
                writer.writerow({
                    'ID Slike': r['image_id'],
                    'Nagib (°)': f"{angle:+.2f}",
                    'Interpretacija': interpretation
                })
        
        print(f"✓ Rezultati su spremljeni u: {OUTPUT_FILE}")
    
    print()


if __name__ == "__main__":
    main()
