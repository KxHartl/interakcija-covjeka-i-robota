"""
Samostalni zadatak 1: Usporedba ground truth i MediaPipe
- Odabrati 5 fotografija iz mape podaci
- Za svaku fotografiju učitati ground truth iz PennAction dataseta
- Odabrati proizvoljnu ključnu točku koju želite usporediti
- Za istu ključnu točku dobiti poziciju pomoću MediaPipe Pose
- Rezultate usporediti i prikazati u tablici
"""

import cv2
import mediapipe as mp
import scipy.io
import numpy as np
import csv
import os
from pathlib import Path

# Konfiguracija
DATA_DIR = 'data/raw/1137'
IMAGE_INDICES = [1, 5, 10, 15, 20]  # koristimo prvi frame iz 1137.mat (000001.jpg)
SELECTED_KEYPOINT = 0 
KEYPOINT_NAMES = {
    0: "Head",
}
# Prikaz slika s označenim GT i MediaPipe točkama
SHOW_IMAGES = True

# Mapiranje PennAction točaka na MediaPipe Pose landmarkove
# PennAction ima 13 točaka, MediaPipe Pose ima 33 landmarka
PENNACTION_TO_MEDIAPIPE = {
    0: mp.solutions.pose.PoseLandmark.NOSE.value,              # Head
}


def load_ground_truth(mat_file_path, image_index):
    """
    Učitaj ground truth podatke iz PennAction .mat datoteke
    
    Args:
        mat_file_path: putanja do .mat datoteke
        image_index: indeks frame-a (0-based)
    
    Returns:
        x, y koordinate za sve ključne točke u tom frame-u
    """
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        x_coords = mat_data['x'][image_index]
        y_coords = mat_data['y'][image_index]
        return x_coords, y_coords
    except Exception as e:
        print(f"Greška pri učitavanju ground truth: {e}")
        return None, None


def get_mediapipe_keypoint(image_path, landmark_idx):
    """
    Detektuj ključnu točku pomoću MediaPipe Pose
    
    Args:
        image_path: putanja do slike
        landmark_idx: indeks landmarka koji trebamo detektovati
    
    Returns:
        tuple: (x_pixel, y_pixel) ili (None, None) ako nije detektirana
    """
    slika = cv2.imread(image_path)
    
    if slika is None:
        print(f"Ne mogu učitati sliku: {image_path}")
        return None, None
    
    h, w = slika.shape[:2]
    
    # MediaPipe Pose inicijalizacija
    mp_pose = mp.solutions.pose
    
    with mp_pose.Pose(static_image_mode=True) as pose:
        slika_rgb = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
        results = pose.process(slika_rgb)
    
    if not results.pose_landmarks:
        print(f"Nije detektirana poza na slici: {image_path}")
        return None, None
    
    # Dohvati specific landmark
    landmark = results.pose_landmarks.landmark[landmark_idx]
    
    # Pretvori iz normaliziranih koordinata u piksele
    x_pixel = int(landmark.x * w)
    y_pixel = int(landmark.y * h)
    
    return x_pixel, y_pixel


def calculate_error(gt_x, gt_y, mp_x, mp_y):
    """
    Izračunaj euklidsku distancu između ground truth i MediaPipe pozicije
    
    Args:
        gt_x, gt_y: ground truth koordinate
        mp_x, mp_y: MediaPipe koordinate
    
    Returns:
        float: euklidska distanca
    """
    if None in [gt_x, gt_y, mp_x, mp_y]:
        return None
    
    return np.sqrt((gt_x - mp_x)**2 + (gt_y - mp_y)**2)


def display_comparison(image_path, gt_point, mp_point, error):
    """
    Prikaži sliku s označenom ground truth i MediaPipe točkom te linijom i tekstom razlike.
    """
    img = cv2.imread(image_path)
    if img is None:
        return

    # Boje
    gt_color = (255, 0, 0)      # plava za ground truth
    mp_color = (0, 0, 255)      # crvena za MediaPipe
    line_color = (0, 255, 255)  # žuta za liniju
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Nacrtaj točke i liniju
    if gt_point is not None:
        cv2.circle(img, (int(gt_point[0]), int(gt_point[1])), 6, gt_color, -1)
        cv2.putText(img, 'GT', (int(gt_point[0]) + 8, int(gt_point[1]) - 8), font, 0.6, gt_color, 2, cv2.LINE_AA)
    if mp_point is not None:
        cv2.circle(img, (int(mp_point[0]), int(mp_point[1])), 6, mp_color, -1)
        cv2.putText(img, 'MP', (int(mp_point[0]) + 8, int(mp_point[1]) - 8), font, 0.6, mp_color, 2, cv2.LINE_AA)
    if gt_point is not None and mp_point is not None:
        cv2.line(img, (int(gt_point[0]), int(gt_point[1])), (int(mp_point[0]), int(mp_point[1])), line_color, 2)

    # Prikaži tekst s razlikom
    info_y = 30
    if error is not None:
        cv2.putText(img, f"Euklidska distanca: {error:.2f} px", (10, info_y), font, 0.7, (0, 128, 0), 2, cv2.LINE_AA)

    # Prikaži sliku i čekaj tipku
    window_name = os.path.basename(image_path)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    """
    Glavna funkcija za usporedbu ground truth i MediaPipe podataka
    """
    
    print("=" * 80)
    print("USPOREDBA GROUND TRUTH I MEDIAPIPE POSE")
    print("=" * 80)
    print(f"\nOdabrana ključna točka: {KEYPOINT_NAMES[SELECTED_KEYPOINT]}")
    print(f"PennAction indeks: {SELECTED_KEYPOINT}")
    print(f"MediaPipe indeks: {PENNACTION_TO_MEDIAPIPE[SELECTED_KEYPOINT]}")
    print("\n" + "=" * 80)
    
    # Učitaj .mat datoteku (koristimo 1137.mat i ground truth za prvi frame)
    mat_file = os.path.join(DATA_DIR, '1137.mat')
    
    results = []
    
    for image_idx in IMAGE_INDICES:
        # Formatiranje imena slike (6 znamenki s vodećim nulama)
        image_name = f"{image_idx:06d}.jpg"
        image_path = os.path.join(DATA_DIR, image_name)
        
        print(f"\nObrađujem sliku: {image_name}")
        
        # Provjera postoji li slika
        if not os.path.exists(image_path):
            print(f"  ⚠️  Slika ne postoji: {image_path}")
            continue
        
        # Učitaj ground truth (indeks je image_idx - 1 jer je 0-based)
        gt_x, gt_y = load_ground_truth(mat_file, image_idx - 1)
        
        if gt_x is None:
            print(f"  ⚠️  Ne mogu učitati ground truth za sliku {image_name}")
            continue
        
        # Pretvori u int ako je float
        gt_x = int(gt_x[SELECTED_KEYPOINT])
        gt_y = int(gt_y[SELECTED_KEYPOINT])
        
        # Detektuj ključnu točku pomoću MediaPipe
        mediapipe_landmark_idx = PENNACTION_TO_MEDIAPIPE[SELECTED_KEYPOINT]
        mp_x, mp_y = get_mediapipe_keypoint(image_path, mediapipe_landmark_idx)
        
        if mp_x is None:
            print(f"  ⚠️  MediaPipe nije detektirao ključnu točku")
            continue
        
        # Izračunaj grešku
        error = calculate_error(gt_x, gt_y, mp_x, mp_y)
        
        # Spremi rezultat
        result = {
            'image_id': image_name,
            'gt_x': gt_x,
            'gt_y': gt_y,
            'mp_x': mp_x,
            'mp_y': mp_y,
            'diff_x': mp_x - gt_x,
            'diff_y': mp_y - gt_y,
            'euclidean_error': error
        }
        results.append(result)
        
        print(f"  ✓ Ground truth: ({gt_x}, {gt_y})")
        print(f"  ✓ MediaPipe:   ({mp_x}, {mp_y})")
        print(f"  ✓ Razlika: ({result['diff_x']:+d}, {result['diff_y']:+d})")
        print(f"  ✓ Euklidska distanca: {error:.2f} px")

        # Prikaži sliku s označenim točkama (ako je omogućeno)
        if SHOW_IMAGES:
            display_comparison(image_path, (gt_x, gt_y), (mp_x, mp_y), error)
    
    # Ispis tablice rezultata
    print("\n" + "=" * 80)
    print("REZULTATI USPOREDBE")
    print("=" * 80)
    print()
    
    if results:
        # Ispis zaglavlja tablice
        print(f"{'ID Slike':<15} {'GT X':<8} {'GT Y':<8} {'MP X':<8} {'MP Y':<8} "
              f"{' ΔX':<8} {' ΔY':<8} {'Greška (px)':<15}")
        print("-" * 100)
        
        # Ispis redaka tablice
        for r in results:
            print(f"{r['image_id']:<15} {r['gt_x']:<8} {r['gt_y']:<8} {r['mp_x']:<8} "
                  f"{r['mp_y']:<8} {r['diff_x']:+<8} {r['diff_y']:+<8} "
                  f"{r['euclidean_error']:<15.2f}")
        
        # Ispis statistike
        print("-" * 100)
        errors = [r['euclidean_error'] for r in results]
        print(f"\n{'STATISTIKA':<15} {'VRIJEDNOST':<15}")
        print("-" * 30)
        print(f"{'Prosječna greška':<15} {np.mean(errors):<15.2f} px")
        print(f"{'Min greška':<15} {np.min(errors):<15.2f} px")
        print(f"{'Max greška':<15} {np.max(errors):<15.2f} px")
        print(f"{'Std dev':<15} {np.std(errors):<15.2f} px")
        print(f"{'Broj slika':<15} {len(results):<15}")
    else:
        print("Nema rezultata za prikaz!")
    
    print("\n" + "=" * 80)
    
    # Spremi rezultate u tekstualnu datoteku
    output_file = 'outputs/body_pose_estimation_01_results.txt'
    os.makedirs('outputs', exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("USPOREDBA GROUND TRUTH I MEDIAPIPE POSE\n")
        f.write("=" * 80 + "\n")
        f.write(f"Odabrana ključna točka: {KEYPOINT_NAMES[SELECTED_KEYPOINT]}\n")
        f.write(f"PennAction indeks: {SELECTED_KEYPOINT}\n")
        f.write(f"MediaPipe indeks: {PENNACTION_TO_MEDIAPIPE[SELECTED_KEYPOINT]}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"{'ID Slike':<15} {'GT X':<8} {'GT Y':<8} {'MP X':<8} {'MP Y':<8} "
                f"{' ΔX':<8} {' ΔY':<8} {'Greška (px)':<15}\n")
        f.write("-" * 100 + "\n")
        
        for r in results:
            f.write(f"{r['image_id']:<15} {r['gt_x']:<8} {r['gt_y']:<8} {r['mp_x']:<8} "
                    f"{r['mp_y']:<8} {r['diff_x']:+<8} {r['diff_y']:+<8} "
                    f"{r['euclidean_error']:<15.2f}\n")
        
        if results:
            f.write("-" * 100 + "\n")
            errors = [r['euclidean_error'] for r in results]
            f.write(f"\nSTATISTIKA\n")
            f.write("-" * 30 + "\n")
            f.write(f"Prosječna greška: {np.mean(errors):.2f} px\n")
            f.write(f"Min greška: {np.min(errors):.2f} px\n")
            f.write(f"Max greška: {np.max(errors):.2f} px\n")
            f.write(f"Std dev: {np.std(errors):.2f} px\n")
            f.write(f"Broj slika: {len(results)}\n")
    
    print(f"✓ Rezultati su spremljeni u: {output_file}")

    # Spremi rezultate i u CSV format
    csv_output_file = 'outputs/body_pose_estimation_01_results.csv'
    with open(csv_output_file, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.writer(csvf)
        # Zaglavlje CSV-a prema zahtjevu
        writer.writerow(['ID Slike','GT X','GT Y','MP X','MP Y','ΔX','ΔY','Greška (px)'])
        for r in results:
            writer.writerow([
                r['image_id'],
                r['gt_x'],
                r['gt_y'],
                r['mp_x'],
                r['mp_y'],
                r['diff_x'],
                r['diff_y'],
                f"{r['euclidean_error']:.2f}" if r['euclidean_error'] is not None else ''
            ])

    print(f"✓ CSV rezultati su spremljeni u: {csv_output_file}")


if __name__ == "__main__":
    main()
