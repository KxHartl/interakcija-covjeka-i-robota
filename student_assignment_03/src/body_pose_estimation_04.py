"""
Samostalni zadatak 4: Brzina gibanja zapešća iz live feed-a
- Koristiti live feed iz kamere
- Pratiti poziciju zapešća u vremenu
- Izračunati brzinu gibanja u m/s
- Spremiti rezultate u .txt datoteku

KOMENTARI:
- Kamera: 30 FPS (tipično)
- Zapešće indeksi: 10 (RIGHT_WRIST), 21 (LEFT_WRIST)
- Brzina: v = distanca / vrijeme
- Pretpostavka: ~30cm od kamere do osobe za kalibraciju
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from collections import deque
from datetime import datetime

# Konfiguracija
COMERA_INDEX = 0  # Kamera laptopa
FPS_BUFFER_SIZE = 30  # Broj frame-ova za izračun FPS-a

# Landmarki zapešća
RIGHT_WRIST = 10
LEFT_WRIST = 21

# Kalibracija: pretpostavljamo da je osoba ~30cm od kamere
# i da zauzima ~60% slike u szerini
CALIBRATION_DISTANCE_CM = 30  # 30cm
CALIBRATION_WIDTH_RATIO = 0.6  # Osoba zauzima ~60% slike


def get_camera_fps(cap):
    """
    Dohvati FPS kamere ili koristi default vrijednost.
    
    Args:
        cap: cv2.VideoCapture objekt
    
    Returns:
        float: FPS vrijednost
    """
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # Default FPS ako kamera ne vraća vrijednost
    return fps


def estimate_pixels_to_cm(frame_width, calibration_distance_cm, calibration_width_ratio):
    """
    Procijeni konverziju piksela u centimetre.
    
    Pretpostavljamo da osoba zauzima određeni dio slike.
    
    Args:
        frame_width: širina frame-a u pikselima
        calibration_distance_cm: procijenja distanca do osobe (cm)
        calibration_width_ratio: omjer koji osoba zauzima u slici
    
    Returns:
        float: konverzijski faktor (cm po pikselu)
    """
    # Prosječna širina ramena ~ 40cm
    shoulder_width_cm = 40
    person_width_pixels = frame_width * calibration_width_ratio
    pixels_per_cm = person_width_pixels / shoulder_width_cm
    return 1.0 / pixels_per_cm  # cm po pikselu


def calculate_distance(point1, point2):
    """
    Izračunaj euklidsku distancu između dvije točke.
    
    Args:
        point1: tuple (x1, y1)
        point2: tuple (x2, y2)
    
    Returns:
        float: distanca u pikselima
    """
    if None in [point1, point2]:
        return 0
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)


def calculate_velocity(distance_pixels, pixels_to_cm, time_delta_sec):
    """
    Izračunaj brzinu na osnovu distanci i vremenske razlike.
    
    Args:
        distance_pixels: distanca u pikselima
        pixels_to_cm: konverzijski faktor
        time_delta_sec: vremenska razlika u sekundama
    
    Returns:
        float: brzina u cm/s ili m/s
    """
    if time_delta_sec <= 0:
        return 0
    
    distance_cm = distance_pixels * pixels_to_cm
    velocity_cm_s = distance_cm / time_delta_sec
    velocity_m_s = velocity_cm_s / 100  # Pretvori u m/s
    
    return velocity_cm_s, velocity_m_s


def get_wrist_position(landmarks, wrist_idx):
    """
    Dohvati poziciju zapešća iz landmarkova.
    
    Args:
        landmarks: lista landmarkova
        wrist_idx: indeks zapešća
    
    Returns:
        tuple: (x, y) pozicija ili None
    """
    if landmarks is None or wrist_idx >= len(landmarks):
        return None
    
    landmark = landmarks[wrist_idx]
    if landmark.visibility < 0.5:  # Ako nije dovoljno vidljivo
        return None
    
    return (landmark.x, landmark.y)


def main():
    """
    Glavna funkcija za praćenje zapešća i mjerenje brzine
    """
    
    print("=" * 90)
    print("PRAĆENJE BRZINE GIBANJA ZAPEŠĆA - LIVE FEED")
    print("=" * 90)
    print(f"\nInicijalizacija kamere...")
    
    # Inicijalizacija kamere
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("❌ Greška: Ne mogu otvoriti kameru!")
        print("Provjeri da li je kamera priključena i dostupna.")
        return
    
    # Dohvati parametre kamere
    fps = get_camera_fps(cap)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    time_per_frame = 1.0 / fps
    
    print(f"✓ Kamera je otvorena")
    print(f"  - Rezolucija: {frame_width}x{frame_height}")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - Vrijeme po frame-u: {time_per_frame*1000:.2f}ms")
    
    # Procijeni konverziju piksela u centimetre
    pixels_to_cm = estimate_pixels_to_cm(
        frame_width,
        CALIBRATION_DISTANCE_CM,
        CALIBRATION_WIDTH_RATIO
    )
    
    print(f"\nKalibracija:")
    print(f"  - Pretpostavljena distanca: {CALIBRATION_DISTANCE_CM}cm")
    print(f"  - Konverzija: {pixels_to_cm:.4f} cm/piksel")
    print(f"  - ili {1/pixels_to_cm:.2f} piksela/cm")
    
    # MediaPipe Pose inicijalizacija
    mp_pose = mp.solutions.pose
    
    # Spremi i aktivne varijable
    previous_wrist_pos = None
    previous_time = time.time()
    
    # Buferi za statistiku
    velocity_buffer = deque(maxlen=FPS_BUFFER_SIZE)
    frame_count = 0
    start_time = time.time()
    
    # Rezultati
    results_data = []
    
    print(f"\n" + "=" * 90)
    print("POČINJE SNIMANJE - Pritisnite 'Q' za izlaz")
    print("=" * 90)
    
    with mp_pose.Pose(
        static_image_mode=False,
        max_num_hands=2,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Greška pri čitanju frame-a")
                break
            
            frame_count += 1
            current_time = time.time()
            time_delta = current_time - previous_time
            
            # Pretvori u RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detektuj pose
            results = pose.process(frame_rgb)
            
            # Inicijalizacija za crtanje
            frame_display = frame.copy()
            right_wrist_pos = None
            left_wrist_pos = None
            right_velocity_m_s = 0
            left_velocity_m_s = 0
            
            if results.pose_landmarks:
                # Desno zapešće
                right_wrist_landmark = results.pose_landmarks.landmark[RIGHT_WRIST]
                if right_wrist_landmark.visibility > 0.5:
                    right_wrist_pos = (
                        int(right_wrist_landmark.x * frame_width),
                        int(right_wrist_landmark.y * frame_height)
                    )
                
                # Lijevo zapešće
                left_wrist_landmark = results.pose_landmarks.landmark[LEFT_WRIST]
                if left_wrist_landmark.visibility > 0.5:
                    left_wrist_pos = (
                        int(left_wrist_landmark.x * frame_width),
                        int(left_wrist_landmark.y * frame_height)
                    )
                
                # Crtanje osnovnih landmarkova
                for landmark in results.pose_landmarks.landmark:
                    h, w, c = frame.shape
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame_display, (cx, cy), 3, (0, 255, 0), -1)
            
            # Izračunaj brzine
            if previous_wrist_pos is not None and right_wrist_pos is not None:
                distance_pixels = calculate_distance(previous_wrist_pos, right_wrist_pos)
                right_velocity_cm_s, right_velocity_m_s = calculate_velocity(
                    distance_pixels, pixels_to_cm, time_delta
                )
            
            # Ispis na frame
            display_text = f"Frame: {frame_count} | FPS: {1.0/time_delta:.1f}"
            cv2.putText(
                frame_display, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            
            if right_wrist_pos is not None:
                cv2.circle(frame_display, right_wrist_pos, 8, (0, 0, 255), -1)  # Crveno za desno
                text_r = f"R: {right_velocity_m_s:.2f} m/s"
                cv2.putText(
                    frame_display, text_r, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )
            
            if left_wrist_pos is not None:
                cv2.circle(frame_display, left_wrist_pos, 8, (255, 0, 0), -1)  # Plavo za lijevo
                text_l = f"L: {left_velocity_m_s:.2f} m/s"
                cv2.putText(
                    frame_display, text_l, (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
                )
            
            # Spremi podatke
            if right_wrist_pos is not None:
                results_data.append({
                    'frame': frame_count,
                    'time': current_time - start_time,
                    'wrist': 'DESNO',
                    'position': right_wrist_pos,
                    'velocity_m_s': right_velocity_m_s
                })
                velocity_buffer.append(right_velocity_m_s)
            
            # Prikaži frame
            cv2.imshow('Praćenje zapešća - Brzina gibanja', frame_display)
            
            # Provjera za izlaz
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\n❌ Prekid snimanja...")
                break
            
            # Ažuriranje pozicije
            previous_wrist_pos = right_wrist_pos if right_wrist_pos is not None else previous_wrist_pos
            previous_time = current_time
    
    # Zatvori kameru
    cap.release()
    cv2.destroyAllWindows()
    
    # Ispis rezultata
    print("\n" + "=" * 90)
    print("REZULTATI SNIMANJA")
    print("=" * 90)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nOpće informacije:")
    print(f"  - Ukupno frame-ova: {frame_count}")
    print(f"  - Ukupno vremena: {elapsed_time:.2f}s")
    print(f"  - Prosječni FPS: {frame_count/elapsed_time:.2f}")
    print(f"  - Broja mjerenja brzine: {len(velocity_buffer)}")
    
    if velocity_buffer:
        velocities = list(velocity_buffer)
        print(f"\nStatistika brzine zapešća:")
        print(f"  - Prosječna brzina: {np.mean(velocities):.3f} m/s")
        print(f"  - Minimalna brzina: {np.min(velocities):.3f} m/s")
        print(f"  - Maksimalna brzina: {np.max(velocities):.3f} m/s")
        print(f"  - Standardna devijacija: {np.std(velocities):.3f} m/s")
    
    print(f"\n" + "=" * 90)
    
    # Spremi rezultate u tekstualnu datoteku
    output_file = 'outputs/body_pose_estimation_04_results.txt'
    os.makedirs('outputs', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 90 + "\n")
        f.write("PRAĆENJE BRZINE GIBANJA ZAPEŠĆA - LIVE FEED\n")
        f.write("=" * 90 + "\n\n")
        
        f.write(f"Datum i vrijeme snimanja: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("PARAMETRI KAMERE:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Rezolucija: {frame_width}x{frame_height}\n")
        f.write(f"FPS: {fps:.2f}\n")
        f.write(f"Vrijeme po frame-u: {time_per_frame*1000:.2f}ms\n\n")
        
        f.write("KALIBRACIJA:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Pretpostavljena distanca: {CALIBRATION_DISTANCE_CM}cm\n")
        f.write(f"Konverzija: {pixels_to_cm:.4f} cm/piksel\n")
        f.write(f"ili {1/pixels_to_cm:.2f} piksela/cm\n\n")
        
        f.write("OPĆE INFORMACIJE:\n")
        f.write("-" * 50 + "\n")
        f.write(f"Ukupno frame-ova: {frame_count}\n")
        f.write(f"Ukupno vremena: {elapsed_time:.2f}s\n")
        f.write(f"Prosječni FPS: {frame_count/elapsed_time:.2f}\n")
        f.write(f"Broj mjerenja brzine: {len(velocity_buffer)}\n\n")
        
        if velocity_buffer:
            velocities = list(velocity_buffer)
            f.write("STATISTIKA BRZINE ZAPEŠĆA:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Prosječna brzina: {np.mean(velocities):.3f} m/s\n")
            f.write(f"Minimalna brzina: {np.min(velocities):.3f} m/s\n")
            f.write(f"Maksimalna brzina: {np.max(velocities):.3f} m/s\n")
            f.write(f"Standardna devijacija: {np.std(velocities):.3f} m/s\n\n")
        
        f.write("DETALJNI PODACI:\n")
        f.write("=" * 90 + "\n")
        f.write(f"{'Frame':<10} {'Vrijeme (s)':<15} {'Zapešće':<10} {'X':<8} {'Y':<8} {'Brzina (m/s)':<15}\n")
        f.write("-" * 90 + "\n")
        
        for data in results_data:
            x, y = data['position']
            f.write(f"{data['frame']:<10} {data['time']:<15.3f} {data['wrist']:<10} "
                   f"{x:<8} {y:<8} {data['velocity_m_s']:<15.3f}\n")
    
    print(f"✓ Detaljni rezultati su spremljeni u: {output_file}")
    print()


if __name__ == "__main__":
    main()
