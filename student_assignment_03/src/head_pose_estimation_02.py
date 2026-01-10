#!/usr/bin/env python3
"""
Head Pose Estimation - Task 2: Head Gaze Direction Classification
Klasifikacija smjera pogleda na temelju Euler kutova iz BIWI dataseta

Zadatak:
- Odabrati 3 fotografija iz mape
- Implementirati jednostavnu klasifikaciju smjera glave prema:
  * Yaw: lijevo/naprijed/desno
  * Pitch: dolje/neutralno/gore
  * Roll: nagib lijevo/bez nagiba/nagib desno
- Rezultate prikazati u tablici

Author: Krešimir Hartl (Student 24)
"""

import numpy as np
import cv2
import math
from pathlib import Path
import pandas as pd

K = np.array([[575.816, 0.0, 320.0],
              [0.0, 575.816, 240.0],
              [0.0,   0.0,   1.0]], dtype=np.float32)


def load_biwi_pose_txt(path: str):
    """
    Load rotation matrix and translation from BIWI pose.txt
    Returns R (3x3) and T in millimetres
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
        # Gimbal lock case (rare)
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0
    
    # Convert from radians to degrees
    return np.degrees([pitch, yaw, roll])


def classify_yaw(yaw: float) -> str:
    """
    Yaw classification:
    - yaw < -10: Osoba gleda ulijevo
    - -10 <= yaw < 10: Osoba gleda naprijed
    - yaw >= 10: Osoba gleda udesno
    """
    if yaw < -10:
        return "Osoba gleda ULIJEVO"
    elif -10 <= yaw < 10:
        return "Osoba gleda NAPRIJED"
    else:  # yaw >= 10
        return "Osoba gleda UDESNO"


def classify_pitch(pitch: float) -> str:
    """
    Pitch classification:
    - pitch < -10: Osoba gleda dolje
    - -10 <= pitch < 10: Osoba gleda neutralno
    - pitch >= 10: Osoba gleda gore
    """
    if pitch < -10:
        return "Osoba gleda DOLJE"
    elif -10 <= pitch < 10:
        return "Osoba gleda NEUTRALNO"
    else:  # pitch >= 10
        return "Osoba gleda GORE"


def classify_roll(roll: float) -> str:
    """
    Roll classification:
    - roll < -10: Nagib glave lijevo
    - -10 <= roll < 10: Nema nagiba glave
    - roll >= 10: Nagib glave desno
    """
    if roll < -10:
        return "Nagib glave LIJEVO"
    elif -10 <= roll < 10:
        return "NEMA nagiba glave"
    else:  # roll >= 10
        return "Nagib glave DESNO"


def classify_head_pose(pitch: float, yaw: float, roll: float) -> str:
    """
    Kombinirana klasifikacija: kombinira sve tri dimenzije
    """
    yaw_class = classify_yaw(yaw)
    pitch_class = classify_pitch(pitch)
    roll_class = classify_roll(roll)
    
    # Kombinirani opis
    combined = f"{yaw_class} | {pitch_class} | {roll_class}"
    return combined


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


def draw_axes_perspective(img, R, T_mm, axis_len_mm=200.0):
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

    return img


def process_frames(frame_ids: list, data_root: Path):
    """
    Process multiple frames and perform head gaze classification.
    """
    results = []
    
    print(f"\nObrada {len(frame_ids)} frame-ova za klasifikaciju smjera pogleda...")
    print("="*140)
    
    for frame_id in frame_ids:
        try:
            rgb_path = data_root / f"frame_{frame_id:05d}_rgb.png"
            pose_path = data_root / f"frame_{frame_id:05d}_pose.txt"
            
            # Load data
            img = cv2.imread(str(rgb_path))
            if img is None:
                print(f"  ✗ Frame {frame_id}: Nije moguće učitati sliku")
                continue
            
            R, T_mm = load_biwi_pose_txt(str(pose_path))
            
            # Calculate Euler angles
            pitch, yaw, roll = rotmat_to_euler_xyz(R)
            
            # Perform classification
            yaw_class = classify_yaw(yaw)
            pitch_class = classify_pitch(pitch)
            roll_class = classify_roll(roll)
            combined_class = classify_head_pose(pitch, yaw, roll)
            
            print(f"\n  ✓ Frame {frame_id}:")
            print(f"    - Pitch: {pitch:7.2f}° → {pitch_class}")
            print(f"    - Yaw:   {yaw:7.2f}° → {yaw_class}")
            print(f"    - Roll:  {roll:7.2f}° → {roll_class}")
            
            # Add to results
            results.append({
                "Frame": frame_id,
                "Pitch (°)": round(pitch, 2),
                "Yaw (°)": round(yaw, 2),
                "Roll (°)": round(roll, 2),
                "Yaw Klasifikacija": yaw_class,
                "Pitch Klasifikacija": pitch_class,
                "Roll Klasifikacija": roll_class,
                "Kombinirana Klasifikacija": combined_class
            })
            
            # Prepare image with visualization
            img_vis = img.copy()
            img_vis = draw_axes_perspective(img_vis, R, T_mm, axis_len_mm=200)
            
            # Add text with classification to image
            y_offset = 30
            cv2.putText(img_vis, f"Frame {frame_id}: Klasifikacija smjera pogleda", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 40
            
            # YAW
            color_yaw = (0, 0, 255)  # Red
            cv2.putText(img_vis, f"YAW: {yaw:6.2f}deg -> {yaw_class}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_yaw, 2)
            y_offset += 35
            
            # PITCH
            color_pitch = (0, 255, 0)  # Green
            cv2.putText(img_vis, f"PITCH: {pitch:6.2f}deg -> {pitch_class}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_pitch, 2)
            y_offset += 35
            
            # ROLL
            color_roll = (255, 0, 0)  # Blue
            cv2.putText(img_vis, f"ROLL: {roll:6.2f}deg -> {roll_class}", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_roll, 2)
            
            # Save image
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            output_img_path = output_dir / f"frame_{frame_id:05d}_classification.png"
            cv2.imwrite(str(output_img_path), img_vis)
            
            # Display image
            cv2.imshow(f"Frame {frame_id} - Klasifikacija", img_vis)
            print(f"     Pritisnite bilo koju tipku za nastavak...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"  ✗ Frame {frame_id}: Greška - {e}")
    
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        
        output_path = Path("outputs") / "head_gaze_classification.csv"
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("\n" + "="*140)
        print("REZULTATI KLASIFIKACIJE SMJERA POGLEDA:")
        print("="*140)
        
        # Display main classification table
        display_df = df[["Frame", "Pitch (°)", "Yaw (°)", "Roll (°)", 
                         "Yaw Klasifikacija", "Pitch Klasifikacija", "Roll Klasifikacija"]].copy()
        print(display_df.to_string(index=False))
        
        print("\n" + "-"*140)
        print("KOMBINIRANA KLASIFIKACIJA (sva tri kuta):")
        print("-"*140)
        for idx, row in df.iterrows():
            print(f"Frame {row['Frame']:3d}: {row['Kombinirana Klasifikacija']}")
        
        print("\n" + "="*140)
        print(f"✓ CSV pohranjen: {output_path}")
        print("="*140 + "\n")
        
        return df
    else:
        print("\n✗ Nema obrađenih frame-ova!")
        return None


if __name__ == "__main__":
    root = Path("./data/raw/hpe/24")
    
    if not root.exists():
        print(f"Greška: Direktorij {root} nije pronađen!")
        exit(1)
    
    # Frames to process - Tri fotografije za klasifikaciju
    frame_ids = [50, 150, 250]
    
    print("\n" + "="*140)
    print("HEAD POSE ESTIMATION - ZADATAK 2")
    print("KLASIFIKACIJA SMJERA POGLEDA")
    print("="*140)
    print("\nKlasifikacijske granice:")
    print("  YAW:   yaw < -10 (ULIJEVO) | -10 <= yaw < 10 (NAPRIJED) | yaw >= 10 (UDESNO)")
    print("  PITCH: pitch < -10 (DOLJE) | -10 <= pitch < 10 (NEUTRALNO) | pitch >= 10 (GORE)")
    print("  ROLL:  roll < -10 (NAGIB LIJEVO) | -10 <= roll < 10 (BEZ NAGIBA) | roll >= 10 (NAGIB DESNO)")
    print("="*140)
    
    # Process frames
    df = process_frames(frame_ids, root)
    
    print("Obrada završena!\n")
