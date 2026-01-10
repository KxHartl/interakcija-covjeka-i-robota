#!/usr/bin/env python3
"""
Head Pose Estimation - Zadatak 3: MediaPipe vs. BIWI
Usporedba orijentacije glave između MediaPipe Face Mesh i BIWI ground truth podataka

Temelji se na:
- head_pose_estimation_01.py (BIWI Euler kutovi)
- head_pose_estimation_02.py (Klasifikacija)
- hpe/04_face_mesh.py (MediaPipe Face Mesh inicijalizacija)

Student: Krešimir Hartl (24)
Reference: vjezbe.pdf - Samostalni zadatak 3
"""

import numpy as np
import cv2
import math
from pathlib import Path
import pandas as pd
import mediapipe as mp

# Intrinzična matrica kamere za BIWI
K = np.array([[575.816, 0.0, 320.0],
              [0.0, 575.816, 240.0],
              [0.0,   0.0,   1.0]], dtype=np.float32)

# Inicijalizacija MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)


def load_biwi_pose_txt(path: str):
    """
    Load rotation matrix and translation from BIWI pose.txt
    Returns R (3x3) and T in millimetres
    
    Prema vjezbe.pdf - 02_pos_rot_info.py
    """
    data = np.loadtxt(path)

    if data.shape == (4, 3):
        R = data[:3, :]
        T_mm = data[3, :]
    elif data.shape == (3, 3):
        R = data
        T_mm = np.zeros(3)
    else:
        raise ValueError(f"Neočekivan oblik pose datoteke: {data.shape}")

    return R.astype(np.float32), T_mm.astype(np.float32)


def rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles (pitch, yaw, roll) in degrees.
    XYZ convention used in BIWI dataset.
    
    Prema vjezbe.pdf:
    - pitch: rotacija oko X osi (glava gore/dolje)
    - yaw: rotacija oko Y osi (glava lijevo/desno)
    - roll: rotacija oko Z osi (nagib glave)
    """
    assert R.shape == (3, 3), f"Expected 3x3 matrix, got {R.shape}"
    
    # Check for gimbal lock
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    
    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])   # rotation around X
        yaw   = math.atan2(-R[2, 0], sy)       # rotation around Y
        roll  = math.atan2(R[1, 0], R[0, 0])   # rotation around Z
    else:
        # Gimbal lock case
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0
    
    return np.degrees([pitch, yaw, roll])


def get_mediapipe_landmarks(image_rgb):
    """
    Pokrenuti MediaPipe Face Mesh i vratiti sve landmarke.
    Povratna vrijednost: lista od 468 (x, y, z) točaka normalizirane ili None
    
    Prema vjezbe.pdf - 04_face_mesh.py
    """
    results = face_mesh.process(image_rgb)
    
    if not results.multi_face_landmarks:
        return None
    
    face = results.multi_face_landmarks[0]
    landmarks = []
    
    for lm in face.landmark:
        landmarks.append((lm.x, lm.y, lm.z))
    
    return landmarks


def denormalize_landmark(lm_norm, image_shape):
    """
    Pretvori normalizirane koordinate (0-1) u pikselske koordinate.
    
    MediaPipe vraća normalizirane koordinate x, y, z
    gdje su x, y u rasponu [0, 1] a z je relativna dubina.
    """
    h, w, _ = image_shape
    x = lm_norm[0] * w
    y = lm_norm[1] * h
    z = lm_norm[2]
    
    return x, y, z


def calculate_head_pose_mediapipe(landmarks, image_shape):
    """
    Izračunati head pose iz MediaPipe Face Mesh landmarka.
    
    Prema vjezbe.pdf - Samostalni zadatak 3:
    - Definirati 3 ključne točke: lijevo oko (lm_index=33), desno oko (lm_index=263), nos (lm_index=1)
    - Izračunati vektore:
      * x-os = vektor između očiju
      * z-os = (trebam izračunati kao normalu na ravnini lica)
      * y-os = z-os × x-os (cross product)
    
    Landmarki iz MediaPipe Face Mesh:
    - 33: Left eye (oko lijevo)
    - 263: Right eye (oko desno)
    - 1: Nose tip (vrh nosa)
    - 168: Forehead (čelo) - mogu koristiti za z-os
    """
    
    if landmarks is None or len(landmarks) < 264:
        return None, None, None
    
    h, w, _ = image_shape
    
    # Definiranje ključnih točaka - prema vjezbe.pdf zadatku 3
    left_eye = np.array(landmarks[33])      # (x_norm, y_norm, z_norm)
    right_eye = np.array(landmarks[263])    # (x_norm, y_norm, z_norm)
    nose = np.array(landmarks[1])           # (x_norm, y_norm, z_norm)
    forehead = np.array(landmarks[168])     # čelo za z-os
    
    # Denormalizacija
    le_px, le_py, le_z = denormalize_landmark(left_eye, image_shape)
    re_px, re_py, re_z = denormalize_landmark(right_eye, image_shape)
    n_px, n_py, n_z = denormalize_landmark(nose, image_shape)
    f_px, f_py, f_z = denormalize_landmark(forehead, image_shape)
    
    # X-os: vektor između očiju (desni - lijevi)
    x_axis = np.array([re_px - le_px, re_py - le_py, re_z - le_z])
    x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    
    # Z-os: vektor od nosa prema čelu (okomita komponenta na ravninu lica)
    # U stvarnosti trebao bi vektor iz lica prema kameri, ali estimiramo iz čela
    z_axis = np.array([f_px - n_px, f_py - n_py, f_z - n_z])
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
    
    # Y-os: cross product z-os × x-os
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
    # Ako je y-os invalidna, recalculate
    if np.linalg.norm(y_axis) < 0.1:
        # Alternativni pristup: z-os je okomita na x-os
        z_axis = np.cross(x_axis, np.array([0, 1, 0]))
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)
    
    # Konstituiranje rotacijske matrice iz osi
    R_mp = np.column_stack([x_axis, y_axis, z_axis])
    
    # Konverzija u Eulerove kutove
    pitch_mp, yaw_mp, roll_mp = rotmat_to_euler_xyz(R_mp)
    
    return pitch_mp, yaw_mp, roll_mp


def project_point(P):
    """
    Project 3D point in camera space to 2D image coordinates.
    P should be in millimetres!
    """
    p = K @ P
    z = p[2]
    if z <= 0.001:
        return 320, 240
    u = p[0] / z
    v = p[1] / z
    return int(round(u)), int(round(v))


def draw_axes_perspective(img, R, T_mm, axis_len_mm=200.0, color_label="BIWI"):
    """
    Draw 3D coordinate axes on image.
    """
    P0 = T_mm
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    PX = T_mm + axis_len_mm * x_axis
    PY = T_mm + axis_len_mm * y_axis
    PZ = T_mm + axis_len_mm * z_axis

    u0, v0 = project_point(P0)
    ux, vx = project_point(PX)
    uy, vy = project_point(PY)
    uz, vz = project_point(PZ)

    cv2.circle(img, (u0, v0), 4, (0, 255, 255), -1)
    cv2.line(img, (u0, v0), (ux, vx), (0, 0, 255), 2)   # X – crvena
    cv2.line(img, (u0, v0), (uy, vy), (0, 255, 0), 2)   # Y – zelena
    cv2.line(img, (u0, v0), (uz, vz), (255, 0, 0), 2)   # Z – plava
    
    # Dodaj labelu
    cv2.putText(img, color_label, (u0 - 30, v0 - 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return img


def draw_mediapipe_landmarks(img, landmarks, image_shape):
    """
    Nacrtati ključne MediaPipe landmarke na slici.
    """
    if landmarks is None:
        return img
    
    h, w, _ = image_shape
    
    # Ključni landmarki za head pose
    key_indices = [33, 263, 1, 168]  # left eye, right eye, nose, forehead
    key_names = ["Left Eye", "Right Eye", "Nose", "Forehead"]
    colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0)]
    
    for idx, name, color in zip(key_indices, key_names, colors):
        lm = landmarks[idx]
        x = int(lm[0] * w)
        y = int(lm[1] * h)
        cv2.circle(img, (x, y), 5, color, -1)
        cv2.putText(img, name, (x + 10, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return img


def process_frames_comparison(frame_ids: list, data_root: Path):
    """
    Obrada frame-ova s usporednom analizom MediaPipe vs BIWI.
    
    Prema vjezbe.pdf - Samostalni zadatak 3:
    1. Za 3 RGB slike iz BIWI dataseta
    2. Pokrenuti MediaPipe Face Mesh
    3. Definirati 3 ključne točke (lijevo oko, desno oko, nos)
    4. Izračunati vektore (x-os, y-os, z-os)
    5. Očitati matricu rotacija ground truth i izračunati Euler kutove
    6. Rezultate usporediti i prikazati u tablici
    """
    results = []
    
    print(f"\nObrada {len(frame_ids)} frame-ova s MediaPipe vs BIWI usporednom analizom...")
    print("="*140)
    
    for frame_id in frame_ids:
        try:
            rgb_path = data_root / f"frame_{frame_id:05d}_rgb.png"
            pose_path = data_root / f"frame_{frame_id:05d}_pose.txt"
            
            # Load RGB slika
            img = cv2.imread(str(rgb_path))
            if img is None:
                print(f"  ✗ Frame {frame_id}: Nije moguće učitati sliku")
                continue
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            
            # BIWI - čitanje ground truth rotacije
            R_biwi, T_mm = load_biwi_pose_txt(str(pose_path))
            pitch_biwi, yaw_biwi, roll_biwi = rotmat_to_euler_xyz(R_biwi)
            
            # MediaPipe - detekcija i izračun
            landmarks_mp = get_mediapipe_landmarks(img_rgb)
            
            if landmarks_mp is None:
                print(f"  ✗ Frame {frame_id}: MediaPipe nije pronašao lice")
                continue
            
            pitch_mp, yaw_mp, roll_mp = calculate_head_pose_mediapipe(landmarks_mp, img.shape)
            
            if pitch_mp is None:
                print(f"  ✗ Frame {frame_id}: Nije moguće izračunati MediaPipe kutove")
                continue
            
            # Izračun razlike
            pitch_diff = abs(pitch_biwi - pitch_mp)
            yaw_diff = abs(yaw_biwi - yaw_mp)
            roll_diff = abs(roll_biwi - roll_mp)
            
            print(f"  ✓ Frame {frame_id}:")
            print(f"     BIWI  - Pitch={pitch_biwi:7.2f}°, Yaw={yaw_biwi:7.2f}°, Roll={roll_biwi:7.2f}°")
            print(f"     MediaPipe - Pitch={pitch_mp:7.2f}°, Yaw={yaw_mp:7.2f}°, Roll={roll_mp:7.2f}°")
            print(f"     Razlika - Pitch={pitch_diff:7.2f}°, Yaw={yaw_diff:7.2f}°, Roll={roll_diff:7.2f}°")
            
            # Dodaj u rezultate
            results.append({
                "Frame": frame_id,
                "Pitch MediaPipe (°)": round(pitch_mp, 2),
                "Pitch BIWI (°)": round(pitch_biwi, 2),
                "Pitch Razlika (°)": round(pitch_diff, 2),
                "Yaw MediaPipe (°)": round(yaw_mp, 2),
                "Yaw BIWI (°)": round(yaw_biwi, 2),
                "Yaw Razlika (°)": round(yaw_diff, 2),
                "Roll MediaPipe (°)": round(roll_mp, 2),
                "Roll BIWI (°)": round(roll_biwi, 2),
                "Roll Razlika (°)": round(roll_diff, 2)
            })
            
            # Vizualizacija
            img_vis = img.copy()
            
            # Nacrtaj MediaPipe landmarke
            img_vis = draw_mediapipe_landmarks(img_vis, landmarks_mp, img_vis.shape)
            
            # Nacrtaj BIWI osi
            img_vis = draw_axes_perspective(img_vis, R_biwi, T_mm, 
                                           axis_len_mm=200, color_label="BIWI GT")
            
            # Dodaj tekstualne informacije
            y_offset = 40
            line_height = 30
            
            cv2.putText(img_vis, "=== BIWI Ground Truth ===", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img_vis, f"Pitch: {pitch_biwi:7.2f}°", (10, y_offset + line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(img_vis, f"Yaw:   {yaw_biwi:7.2f}°", (10, y_offset + 2*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(img_vis, f"Roll:  {roll_biwi:7.2f}°", (10, y_offset + 3*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            
            cv2.putText(img_vis, "=== MediaPipe ===", (10, y_offset + 5*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img_vis, f"Pitch: {pitch_mp:7.2f}°", (10, y_offset + 6*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            cv2.putText(img_vis, f"Yaw:   {yaw_mp:7.2f}°", (10, y_offset + 7*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(img_vis, f"Roll:  {roll_mp:7.2f}°", (10, y_offset + 8*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
            
            cv2.putText(img_vis, "=== Razlika ===", (10, y_offset + 10*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(img_vis, f"ΔPitch: {pitch_diff:6.2f}°", (10, y_offset + 11*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 100, 100), 1)
            cv2.putText(img_vis, f"ΔYaw:   {yaw_diff:6.2f}°", (10, y_offset + 12*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 1)
            cv2.putText(img_vis, f"ΔRoll:  {roll_diff:6.2f}°", (10, y_offset + 13*line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 200), 1)
            
            # Spremi sliku
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            output_img_path = output_dir / f"head_pose_estimation_03_frame_{frame_id:05d}.png"
            cv2.imwrite(str(output_img_path), img_vis)
            
            # Prikaži sliku
            cv2.imshow(f"Frame {frame_id} - MediaPipe vs BIWI", img_vis)
            print(f"     Pritisnite bilo koju tipku za nastavak...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"  ✗ Frame {frame_id}: Greška - {e}")
            import traceback
            traceback.print_exc()
    
    # Spremi u CSV
    if results:
        df = pd.DataFrame(results)
        
        output_path = Path("outputs") / "head_pose_estimation_03.csv"
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("\n" + "="*140)
        print(f"USPOREDNA ANALIZA MediaPipe vs BIWI (pohranjen u head_pose_estimation_03.csv):")
        print("="*140)
        print(df.to_string(index=False))
        print(f"\n✓ CSV pohranjen: {output_path}")
        print("="*140)
        
        return df
    else:
        print("\n✗ Nema obrađenih frame-ova!")
        return None


if __name__ == "__main__":
    root = Path("./data/raw/hpe/24")
    
    if not root.exists():
        print(f"Greška: Direktorij {root} nije pronađen!")
        print(f"Očekivani put: {root}")
        exit(1)
    
    # Frames to process - odabrani iz head_pose_estimation_01.py
    frame_ids = [50, 350, 508]
    
    print("\n" + "="*140)
    print("HEAD POSE ESTIMATION - ZADATAK 3: MediaPipe vs. BIWI - USPOREDBA ORIJENTACIJE GLAVE")
    print("Prema vjezbe.pdf - Samostalni zadatak 3")
    print("Student: Krešimir Hartl (24)")
    print("="*140)
    
    # Process frames with comparison
    df = process_frames_comparison(frame_ids, root)
    
    print("\nObrada završena!\n")
