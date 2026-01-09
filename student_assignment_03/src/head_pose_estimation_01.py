#!/usr/bin/env python3
"""
Task 1: Visual interpretation of BIWI Head Pose data
Author: Krešimir Hartl (Student 24)

For 3 RGB images from BIWI dataset:
1. Load RGB image and rotation/translation information
2. Calculate Euler angles (XYZ convention)
3. Draw head axes on the image
4. Describe where the person is looking
5. Compare results in a table
"""

import numpy as np
import cv2
import math
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd

# Camera intrinsics for BIWI dataset (Kinect)
K = np.array([[575.816, 0.0, 320.0],
              [0.0, 575.816, 240.0],
              [0.0, 0.0, 1.0]], dtype=np.float32)


def load_biwi_pose_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load rotation matrix and translation from BIWI pose.txt file.
    
    Returns:
        R: 3x3 rotation matrix
        T_m: translation in meters (3,)
    """
    data = np.loadtxt(path)
    
    if data.shape != (4, 3):
        raise ValueError(f"Unexpected shape of pose file {path}: {data.shape}")
    
    R = data[:3, :]                 # 3x3 rotation matrix
    T_mm = data[3, :]               # translation in mm
    T_m = T_mm.astype(np.float32) / 1000.0  # convert to meters
    
    return R.astype(np.float32), T_m


def rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles (pitch, yaw, roll).
    Convention: XYZ rotations, angles in degrees
    
    pitch = rotation around X axis (up/down gaze)
    yaw   = rotation around Y axis (left/right gaze)
    roll  = rotation around Z axis (head tilt)
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


def project_point(P: np.ndarray) -> Tuple[int, int]:
    """
    Project 3D point in camera space to 2D image coordinates.
    Uses camera intrinsics matrix K.
    """
    p = K @ P
    u = p[0] / p[2]
    v = p[1] / p[2]
    return int(round(u)), int(round(v))


def draw_axes_on_image(img: np.ndarray, R: np.ndarray, T_mm: np.ndarray,
                       axis_len_mm: float = 200.0) -> np.ndarray:
    """
    Draw 3D coordinate axes on the image based on head pose.
    
    X-axis: red
    Y-axis: green
    Z-axis: blue
    """
    # Convert translation from mm to meters for consistency
    T_m = T_mm / 1000.0
    
    # Head origin
    P0 = T_m
    
    # Local head axes (columns of rotation matrix)
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    
    # Axis endpoints in camera space (meters)
    axis_len_m = axis_len_mm / 1000.0
    PX = T_m + axis_len_m * x_axis
    PY = T_m + axis_len_m * y_axis
    PZ = T_m + axis_len_m * z_axis
    
    # Project to 2D image coordinates
    u0, v0 = project_point(P0)
    ux, vx = project_point(PX)
    uy, vy = project_point(PY)
    uz, vz = project_point(PZ)
    
    # Draw on image
    img_with_axes = img.copy()
    
    # Draw center point
    cv2.circle(img_with_axes, (u0, v0), 5, (0, 255, 255), -1)
    
    # Draw axes
    cv2.line(img_with_axes, (u0, v0), (ux, vx), (0, 0, 255), 3)    # X – red
    cv2.line(img_with_axes, (u0, v0), (uy, vy), (0, 255, 0), 3)    # Y – green
    cv2.line(img_with_axes, (u0, v0), (uz, vz), (255, 0, 0), 3)    # Z – blue
    
    # Add labels
    cv2.putText(img_with_axes, 'X', (ux + 5, vx + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2)
    cv2.putText(img_with_axes, 'Y', (uy + 5, vy + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    cv2.putText(img_with_axes, 'Z', (uz + 5, vz + 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 0, 0), 2)
    
    return img_with_axes


def describe_gaze_direction(pitch: float, yaw: float, roll: float) -> str:
    """
    Generate description of where person is looking based on Euler angles.
    
    Thresholds:
    - pitch < -10°: looking down, pitch > 10°: looking up
    - yaw < -10°: looking left, yaw > 10°: looking right
    - roll < -10°: head tilted left, roll > 10°: head tilted right
    """
    descriptions = []
    
    # Vertical gaze direction (pitch)
    if pitch < -10:
        descriptions.append("gleda dolje")
    elif pitch > 10:
        descriptions.append("gleda gore")
    else:
        descriptions.append("gleda horizontalno")
    
    # Horizontal gaze direction (yaw)
    if yaw < -10:
        descriptions.append("gleda ulijevo")
    elif yaw > 10:
        descriptions.append("gleda udesno")
    else:
        descriptions.append("gleda naprijed")
    
    # Head tilt (roll)
    if roll < -10:
        descriptions.append("nagib ulijevo")
    elif roll > 10:
        descriptions.append("nagib udesno")
    else:
        descriptions.append("bez nagiba")
    
    return ", ".join(descriptions)


def process_frame(frame_num: int, data_root: Path) -> Dict:
    """
    Process a single frame: load image and pose, calculate angles, visualize.
    """
    rgb_path = data_root / f"frame_{frame_num:05d}_rgb.png"
    pose_path = data_root / f"frame_{frame_num:05d}_pose.txt"
    
    # Check files exist
    if not rgb_path.exists() or not pose_path.exists():
        raise FileNotFoundError(f"Files not found for frame {frame_num}")
    
    # Load data
    img = cv2.imread(str(rgb_path))
    if img is None:
        raise ValueError(f"Could not load image: {rgb_path}")
    
    R, T_m = load_biwi_pose_txt(str(pose_path))
    T_mm = T_m * 1000.0  # convert back to mm for visualization
    
    # Calculate Euler angles
    pitch, yaw, roll = rotmat_to_euler_xyz(R)
    
    # Generate gaze description
    gaze_desc = describe_gaze_direction(pitch, yaw, roll)
    
    # Draw axes on image
    img_with_axes = draw_axes_on_image(img, R, T_mm, axis_len_mm=200)
    
    # Add text information to image
    text_info = f"Pitch: {pitch:.1f}°, Yaw: {yaw:.1f}°, Roll: {roll:.1f}°"
    cv2.putText(img_with_axes, text_info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    
    # Save result
    output_path = data_root.parent.parent.parent / f"outputs/frame_{frame_num:05d}_with_axes.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_with_axes)
    print(f"Saved: {output_path}")
    
    return {
        "Frame": frame_num,
        "Pitch (°)": f"{pitch:.2f}",
        "Yaw (°)": f"{yaw:.2f}",
        "Roll (°)": f"{roll:.2f}",
        "Gaze Direction": gaze_desc,
        "BIWI Accuracy": "Good"
    }


def main():
    """
    Main function: Process 3 frames and create results table.
    """
    # Data directory for student 24 (Krešimir Hartl)
    data_root = Path("data/raw/hpe/24")
    
    if not data_root.exists():
        print(f"Error: Data directory not found at {data_root}")
        print("Please ensure the BIWI dataset is in the correct location.")
        return
    
    # Frames to process
    frame_ids = [50, 150, 250]
    
    print("="*80)
    print("Task 1: Visual Interpretation of BIWI Head Pose Data")
    print("Student: Krešimir Hartl (24)")
    print("="*80)
    
    results = []
    
    for frame_id in frame_ids:
        try:
            print(f"\nProcessing frame {frame_id}...")
            result = process_frame(frame_id, data_root)
            results.append(result)
            print(f"✓ Frame {frame_id} completed successfully")
        except Exception as e:
            print(f"✗ Error processing frame {frame_id}: {e}")
    
    # Create results table
    if results:
        df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("RESULTS TABLE")
        print("="*80)
        print(df.to_string(index=False))
        
        # Save table to CSV
        output_csv = Path("outputs/task1_results.csv")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nTable saved to: {output_csv}")
    
    print("\n" + "="*80)
    print("Processing complete!")
    print("="*80)


if __name__ == "__main__":
    main()
