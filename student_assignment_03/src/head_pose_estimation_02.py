#!/usr/bin/env python3
"""
Head Pose Estimation - Zadatak 2: Klasifikacija smjera pogleda
Temelji se na head_pose_estimation_01.py

Klasifikacija smjera glave prema Eulerovim kutovima:
- YAW: gleda lijevo/naprijed/desno
- PITCH: gleda dolje/neutralno/gore
- ROLL: nagib glave lijevo/nema nagiba/desno

Author: Krešimir Hartl (Student 24)
Reference: vjezbe.pdf - Head pose estimation klasifikacija
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

    return R.astype(np.float32), T_mm.astype(np.float32)  # T u mm


def rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles (pitch, yaw, roll) in degrees.
    XYZ convention used in BIWI dataset.
    
    Prema dogovoru iz vjezbe.pdf:
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
        # Gimbal lock case (rare)
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0
    
    # Convert from radians to degrees
    return np.degrees([pitch, yaw, roll])


def classify_gaze_direction(pitch: float, yaw: float, roll: float) -> dict:
    """
    Klasificira smjer pogleda prema Eulerovim kutovima.
    
    Tablica iz vjezbe.pdf - Samostalni zadatak 2:
    
    YAW klasifikacija:
    - yaw < -10: Osoba gleda ulijevo
    - -10 <= yaw < 10: Osoba gleda naprijed
    - yaw > 10: Osoba gleda udesno
    
    PITCH klasifikacija:
    - pitch < -10: Osoba gleda dolje
    - -10 <= pitch < 10: Osoba gleda neutralno
    - pitch > 10: Osoba gleda gore
    
    ROLL klasifikacija:
    - roll < -10: Nagib glave lijevo
    - -10 <= roll < 10: Nema nagiba glave
    - roll > 10: Nagib glave desno
    """
    
    # YAW klasifikacija
    if yaw < -10:
        yaw_class = "Gleda ulijevo"
    elif -10 <= yaw < 10:
        yaw_class = "Gleda naprijed"
    else:  # yaw >= 10
        yaw_class = "Gleda udesno"
    
    # PITCH klasifikacija
    if pitch < -10:
        pitch_class = "Gleda gore"
    elif -10 <= pitch < 10:
        pitch_class = "Gleda neutralno"
    else:  # pitch >= 10
        pitch_class = "Gleda dolje"
    
    # ROLL klasifikacija
    if roll < -10:
        roll_class = "Nagib glave desno"
    elif -10 <= roll < 10:
        roll_class = "Nema nagiba glave"
    else:  # roll >= 10
        roll_class = "Nagib glave lijevo"
    
    return {
        "yaw_class": yaw_class,
        "pitch_class": pitch_class,
        "roll_class": roll_class,
        "combined_description": f"{pitch_class}, {yaw_class}, {roll_class}"
    }


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
    # ishodište (glava)
    P0 = T_mm

    # lokalne osi glave
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


def draw_classification_text(img, pitch: float, yaw: float, roll: float, 
                            classification: dict, margin_left=10, margin_top=40, 
                            line_height=30, font_size=0.7):
    """
    Draw classification results on image with structured text layout.
    """
    cv2.putText(img, f"=== KUTOVI ===", 
               (margin_left, margin_top), 
               cv2.FONT_HERSHEY_SIMPLEX, font_size+0.1, (255, 255, 0), 2)
    
    cv2.putText(img, f"Pitch: {pitch:7.2f}°", 
               (margin_left, margin_top + line_height), 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 1)
    
    cv2.putText(img, f"Yaw:   {yaw:7.2f}°", 
               (margin_left, margin_top + 2*line_height), 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 1)
    
    cv2.putText(img, f"Roll:  {roll:7.2f}°", 
               (margin_left, margin_top + 3*line_height), 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 1)
    
    cv2.putText(img, f"=== KLASIFIKACIJA ===", 
               (margin_left, margin_top + 5*line_height), 
               cv2.FONT_HERSHEY_SIMPLEX, font_size+0.1, (255, 255, 0), 2)
    
    cv2.putText(img, classification["pitch_class"], 
               (margin_left, margin_top + 6*line_height), 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 255), 1)
    
    cv2.putText(img, classification["yaw_class"], 
               (margin_left, margin_top + 7*line_height), 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 255, 0), 1)
    
    cv2.putText(img, classification["roll_class"], 
               (margin_left, margin_top + 8*line_height), 
               cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), 1)
    
    return img


def process_frames_with_classification(frame_ids: list, data_root: Path):
    """
    Process multiple frames and classify gaze direction.
    Extract results to CSV with detailed classification.
    """
    results = []
    
    print(f"\nObrada {len(frame_ids)} frame-ova s klasifikacijom smjera pogleda...")
    print("="*120)
    
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
            
            # Classify gaze direction
            classification = classify_gaze_direction(pitch, yaw, roll)
            
            print(f"  ✓ Frame {frame_id}:")
            print(f"     Kutovi: Pitch={pitch:7.2f}°, Yaw={yaw:7.2f}°, Roll={roll:7.2f}°")
            print(f"     {classification['combined_description']}")
            
            # Add to results
            results.append({
                "Frame": frame_id,
                "Pitch (°)": round(pitch, 2),
                "Yaw (°)": round(yaw, 2),
                "Roll (°)": round(roll, 2),
                "Pitch Klasifikacija": classification["pitch_class"],
                "Yaw Klasifikacija": classification["yaw_class"],
                "Roll Klasifikacija": classification["roll_class"]
            })
            
            # Draw axes on image
            img_vis = draw_axes_perspective(img.copy(), R, T_mm, axis_len_mm=200)
            
            # Draw classification text
            img_vis = draw_classification_text(img_vis, pitch, yaw, roll, classification)
            
            # Save image
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            output_img_path = output_dir / f"head_pose_estimation_02_frame_{frame_id:05d}.png"
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
        
        output_path = Path("outputs") / "head_pose_estimation_02.csv"
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("\n" + "="*120)
        print(f"REZULTATI KLASIFIKACIJE (pohranjen u head_pose_estimation_02.csv):")
        print("="*120)
        print(df.to_string(index=False))
        print(f"\n✓ CSV pohranjen: {output_path}")
        print("="*120)
        
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
    
    print("\n" + "="*120)
    print("HEAD POSE ESTIMATION - ZADATAK 2: KLASIFIKACIJA SMJERA POGLEDA")
    print("Temelj: BIWI dataset + Euler kutovi (XYZ konvencija)")
    print("Student: Krešimir Hartl (24)")
    print("Reference: vjezbe.pdf - Samostalni zadatak 2")
    print("="*120)
    
    # Process frames with classification
    df = process_frames_with_classification(frame_ids, root)
    
    print("\nObrada završena!\n")
