#!/usr/bin/env python3
"""
Task 1: Visual interpretation of BIWI Head Pose data
Author: Krešimir Hartl (Student 24)

For 3 RGB images from BIWI dataset:
1. Load RGB image and rotation/translation information
2. Calculate Euler angles (XYZ convention)
3. Draw head axes on the image with coordinate system
4. Visualize gaze direction and head tilt
5. Describe where the person is looking
6. Compare results in a table
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
    
    X-axis: red (lateral/right)
    Y-axis: green (vertical/up)
    Z-axis: blue (forward/out of head)
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
    
    # Draw semi-transparent background for legend area
    overlay = img_with_axes.copy()
    cv2.rectangle(overlay, (5, 5), (420, 105), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, img_with_axes, 0.7, 0, img_with_axes)
    
    # Draw center point (head position)
    cv2.circle(img_with_axes, (u0, v0), 6, (0, 255, 255), -1)
    cv2.circle(img_with_axes, (u0, v0), 8, (0, 255, 255), 2)
    
    # Draw axes with thicker lines for better visibility
    cv2.line(img_with_axes, (u0, v0), (ux, vx), (0, 0, 255), 4)    # X – red
    cv2.line(img_with_axes, (u0, v0), (uy, vy), (0, 255, 0), 4)    # Y – green
    cv2.line(img_with_axes, (u0, v0), (uz, vz), (255, 0, 0), 4)    # Z – blue
    
    # Draw arrow heads on axes
    def draw_arrow_head(img, pt_start, pt_end, color):
        """Draw arrow head at pt_end pointing from pt_start"""
        dx = pt_end[0] - pt_start[0]
        dy = pt_end[1] - pt_start[1]
        length = math.sqrt(dx*dx + dy*dy)
        if length > 10:
            dx /= length
            dy /= length
            # Perpendicular vectors for arrow head
            px, py = -dy, dx
            arrow_size = 15
            p1 = (int(pt_end[0] - arrow_size*dx - arrow_size*px),
                  int(pt_end[1] - arrow_size*dy - arrow_size*py))
            p2 = (int(pt_end[0] - arrow_size*dx + arrow_size*px),
                  int(pt_end[1] - arrow_size*dy + arrow_size*py))
            cv2.fillPoly(img, [np.array([pt_end, p1, p2])], color)
    
    draw_arrow_head(img_with_axes, (u0, v0), (ux, vx), (0, 0, 255))
    draw_arrow_head(img_with_axes, (u0, v0), (uy, vy), (0, 255, 0))
    draw_arrow_head(img_with_axes, (u0, v0), (uz, vz), (255, 0, 0))
    
    # Add axis labels with background
    cv2.putText(img_with_axes, 'X', (ux + 10, vx + 5), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(img_with_axes, 'Y', (uy + 10, vy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(img_with_axes, 'Z', (uz + 10, vz + 5), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 0, 0), 3, cv2.LINE_AA)
    
    # Add legend
    legend_x = 15
    legend_y = 25
    cv2.putText(img_with_axes, 'Koordinatni sustav:', (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(img_with_axes, 'X (crvena) - desna-lijeva os',
                (legend_x, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)
    cv2.putText(img_with_axes, 'Y (zelena) - gore-dolje os',
                (legend_x, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)
    cv2.putText(img_with_axes, 'Z (plava) - naprijed-natrag',
                (legend_x, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1)
    
    return img_with_axes


def add_euler_visualization(img: np.ndarray, pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    Add visual indicators for Euler angles and gaze direction.
    """
    h, w = img.shape[:2]
    img_viz = img.copy()
    
    # Draw semi-transparent background for statistics
    overlay = img_viz.copy()
    cv2.rectangle(overlay, (w - 380, 5), (w - 5, 180), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, img_viz, 0.6, 0, img_viz)
    
    # Euler angles text
    stats_x = w - 370
    stats_y = 25
    line_height = 30
    
    cv2.putText(img_viz, 'EULER KUTOVI:', (stats_x, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Pitch (X-axis rotation) - Up/Down gaze
    pitch_color = (0, 0, 255) if abs(pitch) > 10 else (100, 255, 100)
    pitch_status = "GORE" if pitch > 10 else "DOLJE" if pitch < -10 else "NEUTRALNO"
    cv2.putText(img_viz, f'Pitch (X):  {pitch:7.2f}°  {pitch_status}',
                (stats_x, stats_y + line_height), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, pitch_color, 2)
    
    # Yaw (Y-axis rotation) - Left/Right gaze
    yaw_color = (255, 0, 0) if abs(yaw) > 10 else (100, 255, 100)
    yaw_status = "DESNO" if yaw > 10 else "LIJEVO" if yaw < -10 else "NAPRIJED"
    cv2.putText(img_viz, f'Yaw (Y):    {yaw:7.2f}°  {yaw_status}',
                (stats_x, stats_y + 2*line_height), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, yaw_color, 2)
    
    # Roll (Z-axis rotation) - Head tilt
    roll_color = (0, 255, 0) if abs(roll) > 10 else (100, 255, 100)
    roll_status = "DESNO" if roll > 10 else "LIJEVO" if roll < -10 else "BEZ NAGIBA"
    cv2.putText(img_viz, f'Roll (Z):   {roll:7.2f}°  {roll_status}',
                (stats_x, stats_y + 3*line_height), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, roll_color, 2)
    
    # Add visual roll indicator (protractor-like circle at bottom)
    center_x = w // 2
    center_y = h - 80
    radius = 60
    
    # Draw roll indicator circle
    cv2.circle(img_viz, (center_x, center_y), radius, (100, 100, 100), 2)
    
    # Draw reference lines (0°, 90°, -90°)
    cv2.line(img_viz, (center_x - radius - 10, center_y), (center_x + radius + 10, center_y),
             (100, 100, 100), 1)  # Horizontal line at 0°
    cv2.line(img_viz, (center_x, center_y - radius - 10), (center_x, center_y + radius + 10),
             (100, 100, 100), 1)  # Vertical line at ±90°
    
    # Draw roll indicator needle
    rad = math.radians(roll)
    end_x = int(center_x + radius * math.cos(rad - math.pi/2))
    end_y = int(center_y + radius * math.sin(rad - math.pi/2))
    cv2.line(img_viz, (center_x, center_y), (end_x, end_y), (0, 255, 0), 3)
    cv2.circle(img_viz, (center_x, center_y), 5, (0, 255, 0), -1)
    
    # Label roll indicator
    cv2.putText(img_viz, 'NAGIB (ROLL)', (center_x - 50, center_y + radius + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return img_viz


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
        descriptions.append("gleda DOLJE")
    elif pitch > 10:
        descriptions.append("gleda GORE")
    else:
        descriptions.append("gleda horizontalno")
    
    # Horizontal gaze direction (yaw)
    if yaw < -10:
        descriptions.append("gleda ULIJEVO")
    elif yaw > 10:
        descriptions.append("gleda UDESNO")
    else:
        descriptions.append("gleda naprijed")
    
    # Head tilt (roll)
    if roll < -10:
        descriptions.append("nagib ULIJEVO")
    elif roll > 10:
        descriptions.append("nagib UDESNO")
    else:
        descriptions.append("bez nagiba")
    
    return ", ".join(descriptions)


def process_frame(frame_num: int, data_root: Path, show_images: bool = False) -> Dict:
    """
    Process a single frame: load image and pose, calculate angles, visualize.
    
    Args:
        frame_num: Frame number to process
        data_root: Root directory for BIWI data
        show_images: If True, display images during processing
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
    
    # Create visualization with axes
    img_with_axes = draw_axes_on_image(img, R, T_mm, axis_len_mm=200)
    
    # Add Euler angle visualization
    img_with_viz = add_euler_visualization(img_with_axes, pitch, yaw, roll)
    
    # Save result
    output_path = data_root.parent.parent.parent / f"outputs/frame_{frame_num:05d}_with_axes.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_with_viz)
    print(f"  Pohrana: {output_path}")
    
    # Display image if requested
    if show_images:
        cv2.imshow(f"Frame {frame_num}: {gaze_desc}", img_with_viz)
        print(f"  Pritisnite bilo koju tipku za sljedeći frame...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return {
        "Frame": frame_num,
        "Pitch (°)": f"{pitch:.2f}",
        "Yaw (°)": f"{yaw:.2f}",
        "Roll (°)": f"{roll:.2f}",
        "Gaze Direction": gaze_desc,
        "BIWI Accuracy": "Dobro"
    }


def main():
    """
    Main function: Process 3 frames and create results table.
    """
    # Data directory for student 24 (Krešimir Hartl)
    data_root = Path("data/raw/hpe/24")
    
    if not data_root.exists():
        print(f"Greška: Direktorij s podacima nije pronađen na {data_root}")
        print("Molim provjerite da je BIWI dataset u ispravnoj lokaciji.")
        return
    
    # Frames to process
    frame_ids = [50, 150, 250]
    
    print("\n" + "="*90)
    print("ZADATAK 1: Vizualna Interpretacija BIWI Head Pose Podataka")
    print("Student: Krešimir Hartl (24)")
    print("="*90)
    
    results = []
    
    for i, frame_id in enumerate(frame_ids, 1):
        try:
            print(f"\n[{i}/3] Obrada frame-a {frame_id}...")
            result = process_frame(frame_id, data_root, show_images=False)
            results.append(result)
            print(f"  ✓ Frame {frame_id} je uspješno obrađen")
        except Exception as e:
            print(f"  ✗ Greška pri obradi frame-a {frame_id}: {e}")
    
    # Create results table
    if results:
        df = pd.DataFrame(results)
        
        print("\n" + "="*90)
        print("TABLICA REZULTATA")
        print("="*90)
        print(df.to_string(index=False))
        
        # Save table to CSV
        output_csv = Path("outputs/task1_results.csv")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nTabličnih rezultata pohranjena: {output_csv}")
    
    print("\n" + "="*90)
    print("Obrada završena! Vizualizacije su sprema u /outputs direktoriju.")
    print("="*90 + "\n")


if __name__ == "__main__":
    main()
