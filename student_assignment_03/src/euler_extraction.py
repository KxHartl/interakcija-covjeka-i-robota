#!/usr/bin/env python3
"""
Minimal modification of 03_vis_pos_rot_info.py
Extracts Euler angles from BIWI dataset and saves to CSV
Adds image visualization with axes

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

    return R.astype(np.float32), T_mm.astype(np.float32)  # T u mm


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
    Minimal modification of original function from 03_vis_pos_rot_info.py
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


def process_frames(frame_ids: list, data_root: Path):
    """
    Process multiple frames and extract Euler angles to CSV.
    Also display images with axes.
    """
    results = []
    
    print(f"\nObrada {len(frame_ids)} frame-ova...")
    print("="*100)
    
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
            
            print(f"  ✓ Frame {frame_id}: Pitch={pitch:.2f}°, Yaw={yaw:.2f}°, Roll={roll:.2f}°")
            
            # Add to results
            results.append({
                "Frame": frame_id,
                "Pitch (°)": round(pitch, 2),
                "Yaw (°)": round(yaw, 2),
                "Roll (°)": round(roll, 2),
                "T_X (mm)": round(T_mm[0], 1),
                "T_Y (mm)": round(T_mm[1], 1),
                "T_Z (mm)": round(T_mm[2], 1)
            })
            
            # Draw axes on image (minimal modification)
            img_axes = draw_axes_perspective(img.copy(), R, T_mm, axis_len_mm=200)
            
            # Add text with angles to image
            cv2.putText(img_axes, f"Pitch: {pitch:.2f}deg", (10, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(img_axes, f"Yaw: {yaw:.2f}deg", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img_axes, f"Roll: {roll:.2f}deg", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # Save image
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            output_img_path = output_dir / f"frame_{frame_id:05d}_euler.png"
            cv2.imwrite(str(output_img_path), img_axes)
            
            # Display image
            cv2.imshow(f"Frame {frame_id} - Euler kutovi", img_axes)
            print(f"     Pritisnite bilo koju tipku za nastavak...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        except Exception as e:
            print(f"  ✗ Frame {frame_id}: Greška - {e}")
    
    # Save to CSV
    if results:
        df = pd.DataFrame(results)
        
        output_path = Path("outputs") / "euler_angles.csv"
        output_path.parent.mkdir(exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print("\n" + "="*100)
        print(f"REZULTATI (pohranjen u euler_angles.csv):")
        print("="*100)
        print(df.to_string(index=False))
        print(f"\n✓ CSV pohranjen: {output_path}")
        print("="*100)
        
        return df
    else:
        print("\n✗ Nema obrađenih frame-ova!")
        return None


if __name__ == "__main__":
    root = Path("./data/raw/hpe/24")
    
    if not root.exists():
        print(f"Greška: Direktorij {root} nije pronađen!")
        exit(1)
    
    # Frames to process
    frame_ids = [50, 150, 250]
    
    print("\n" + "="*100)
    print("EKSTRAKCIJA EULER KUTOVA IZ BIWI DATASETA")
    print("Minimal modificirana verzija 03_vis_pos_rot_info.py")
    print("Student: Krešimir Hartl (24)")
    print("="*100)
    
    # Process frames
    df = process_frames(frame_ids, root)
    
    print("\nObrada završena!\n")
