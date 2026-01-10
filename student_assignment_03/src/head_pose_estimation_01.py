#!/usr/bin/env python3
"""
Task 1: Visual interpretation of BIWI Head Pose data
Author: Krešimir Hartl (Student 24)

Integrated features from src/hpe utilities:
1. Depth data decoding and visualization (01_depth_info.py)
2. Position and rotation loading (02_pos_rot_info.py)
3. 3D perspective axes visualization (03_vis_pos_rot_info.py)
4. Face mesh landmarks from MediaPipe (04_face_mesh.py)
5. Enhanced Euler angle visualization and roll gauge

CALIBRATION FIX v3:
- Added camera origin (0,0,0) as reference point
- Shows both where axes start and where camera is located
- T is displayed relative to camera
"""

import numpy as np
import cv2
import math
from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
import mediapipe as mp

# Camera intrinsics for BIWI dataset (Kinect)
K = np.array([[575.816, 0.0, 320.0],
              [0.0, 575.816, 240.0],
              [0.0, 0.0, 1.0]], dtype=np.float32)

# MediaPipe FaceMesh initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# ============================================================================
# UTILITY FUNCTIONS FROM src/hpe
# ============================================================================

def decode_biwi_depth(filepath: str) -> np.ndarray:
    """
    Decode BIWI depth binary format (from 01_depth_info.py)
    Returns depth in millimetres as uint16
    """
    with open(filepath, 'rb') as f:
        width = int.from_bytes(f.read(4), 'little')
        height = int.from_bytes(f.read(4), 'little')

        depth = np.zeros([width * height], dtype=np.uint16)
        i = 0

        while i < width * height:
            skip = int.from_bytes(f.read(4), 'little')
            read = int.from_bytes(f.read(4), 'little')

            for _ in range(read):
                depth[i + skip] = int.from_bytes(f.read(2), 'little')
                i += 1

            i += skip

    depth = depth.reshape((height, width))

    # Add border and resize to 640x480 (BIWI specific)
    d_ = np.zeros((depth.shape[0] + 20, depth.shape[1] + 20), dtype=np.uint16)
    d_[20:, :640] = depth
    depth = cv2.resize(d_, (640, 480), interpolation=cv2.INTER_NEAREST)

    return depth


def load_biwi_pose_txt(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load rotation matrix and translation from BIWI pose.txt file (from 02_pos_rot_info.py).
    
    Returns:
        R: 3x3 rotation matrix
        T: translation vector as stored in file (likely in mm)
    """
    data = np.loadtxt(path)
    
    if data.shape != (4, 3):
        raise ValueError(f"Unexpected shape of pose file {path}: {data.shape}")
    
    R = data[:3, :]     # 3x3 rotation matrix
    T = data[3, :]      # translation vector
    
    return R.astype(np.float32), T.astype(np.float32)


def rotmat_to_euler_xyz(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to Euler angles (pitch, yaw, roll) (from 02_pos_rot_info.py).
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
    Project 3D point in camera space to 2D image coordinates (from 03_vis_pos_rot_info.py).
    Uses camera intrinsics matrix K.
    
    P: Point in 3D camera space
    Returns: (u, v) image coordinates or (320, 240) if behind camera
    """
    p = K @ P
    z = p[2]
    if z <= 0.001:  # Behind or too close to camera
        return 320, 240  # Return image center as fallback
    u = p[0] / z
    v = p[1] / z
    return int(round(u)), int(round(v))


def get_face_landmarks(img: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extract 468 face landmarks using MediaPipe FaceMesh (from 04_face_mesh.py)
    """
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    
    landmarks = []
    if results.multi_face_landmarks:
        h, w, _ = img.shape
        for face in results.multi_face_landmarks:
            for lm in face.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                landmarks.append((x, y))
    
    return landmarks


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def draw_axes_on_image(img: np.ndarray, R: np.ndarray, T: np.ndarray,
                       axis_len: float = 200.0) -> np.ndarray:
    """
    Draw 3D coordinate axes on the image based on head pose (from 03_vis_pos_rot_info.py).
    
    Now with dual reference points:
    - RED CROSS: Camera origin (0, 0, 0) - reference point
    - CYAN CIRCLE: Head center - where T points to
    
    X-axis: red (lateral/right)
    Y-axis: green (vertical/up)
    Z-axis: blue (forward/out of head)
    """
    # T represents translation from camera to head (head position in camera space)
    P0 = T.astype(np.float32)

    # Local head axes (columns of rotation matrix)
    x_axis = R[:, 0].astype(np.float32)
    y_axis = R[:, 1].astype(np.float32)
    z_axis = R[:, 2].astype(np.float32)

    # Axis endpoints in camera space
    PX = P0 + axis_len * x_axis
    PY = P0 + axis_len * y_axis
    PZ = P0 + axis_len * z_axis
    
    # Project to 2D image coordinates
    u0, v0 = project_point(P0)
    ux, vx = project_point(PX)
    uy, vy = project_point(PY)
    uz, vz = project_point(PZ)
    
    # Also project camera origin for reference
    camera_origin = np.array([0, 0, 0.001], dtype=np.float32)  # Use small z to avoid division by zero
    u_cam, v_cam = project_point(camera_origin)
    
    # Draw on image
    img_with_axes = img.copy()
    
    # Draw semi-transparent background for legend area
    overlay = img_with_axes.copy()
    cv2.rectangle(overlay, (5, 5), (450, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, img_with_axes, 0.7, 0, img_with_axes)
    
    # Draw camera origin as reference (RED CROSS)
    cross_size = 15
    cv2.line(img_with_axes, (u_cam - cross_size, v_cam), (u_cam + cross_size, v_cam),
             (0, 0, 255), 2)  # Horizontal
    cv2.line(img_with_axes, (u_cam, v_cam - cross_size), (u_cam, v_cam + cross_size),
             (0, 0, 255), 2)  # Vertical
    cv2.putText(img_with_axes, 'CAM', (u_cam + 10, v_cam - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 255), 1)
    
    # Draw head center point (CYAN CIRCLE)
    cv2.circle(img_with_axes, (u0, v0), 8, (0, 255, 255), -1)
    cv2.circle(img_with_axes, (u0, v0), 10, (0, 255, 255), 2)
    
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
    
    # Add axis labels
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
    cv2.putText(img_with_axes, 'X (crvena) - desna-lijeva',
                (legend_x, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)
    cv2.putText(img_with_axes, 'Y (zelena) - gore-dolje',
                (legend_x, legend_y + 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 0), 1)
    cv2.putText(img_with_axes, 'Z (plava) - naprijed-natrag',
                (legend_x, legend_y + 65), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 0, 0), 1)
    cv2.putText(img_with_axes, 'CAM (crveni X) - kamera (0,0,0)',
                (legend_x, legend_y + 85), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)
    cv2.putText(img_with_axes, f'T = ({T[0]:.0f}, {T[1]:.0f}, {T[2]:.0f})',
                (legend_x, legend_y + 105), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (180, 180, 180), 1)
    
    return img_with_axes


def add_euler_visualization(img: np.ndarray, pitch: float, yaw: float, roll: float) -> np.ndarray:
    """
    Add visual indicators for Euler angles and gaze direction with roll gauge.
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


def draw_face_mesh(img: np.ndarray) -> np.ndarray:
    """
    Draw 468 face mesh landmarks using MediaPipe (from 04_face_mesh.py)
    """
    landmarks = get_face_landmarks(img)
    img_mesh = img.copy()
    
    # Draw all landmarks
    for (x, y) in landmarks:
        cv2.circle(img_mesh, (x, y), 1, (0, 255, 0), -1)
    
    # Add label
    cv2.putText(img_mesh, f'Face Landmarks: {len(landmarks)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return img_mesh


def visualize_depth_map(depth_mm: np.ndarray) -> np.ndarray:
    """
    Visualize depth map with proper normalization (from 01_depth_info.py)
    """
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid = depth_m[depth_m > 0]
    
    if valid.size > 0:
        # Normalize depth for visualization
        depth_vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)
        
        # Apply colormap for better visualization
        depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        return depth_color
    else:
        # Return black image if no valid depth data
        return np.zeros_like(depth_mm, dtype=np.uint8)


def describe_gaze_direction(pitch: float, yaw: float, roll: float) -> str:
    """
    Generate description of where person is looking based on Euler angles.
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
    Process a single frame: load image, depth, and pose data, calculate angles, visualize.
    
    Args:
        frame_num: Frame number to process
        data_root: Root directory for BIWI data
        show_images: If True, display images during processing
    """
    rgb_path = data_root / f"frame_{frame_num:05d}_rgb.png"
    pose_path = data_root / f"frame_{frame_num:05d}_pose.txt"
    depth_path = data_root / f"frame_{frame_num:05d}_depth.bin"
    
    # Check files exist
    if not rgb_path.exists() or not pose_path.exists():
        raise FileNotFoundError(f"Files not found for frame {frame_num}")
    
    # Load RGB data
    img = cv2.imread(str(rgb_path))
    if img is None:
        raise ValueError(f"Could not load image: {rgb_path}")
    
    # Load pose data
    R, T = load_biwi_pose_txt(str(pose_path))
    
    # Load depth data if available
    depth_mm = None
    depth_info = "N/A"
    if depth_path.exists():
        try:
            depth_mm = decode_biwi_depth(str(depth_path))
            valid_depths = depth_mm[depth_mm > 0]
            if valid_depths.size > 0:
                depth_info = f"{valid_depths.min()} - {valid_depths.max()} mm"
        except Exception as e:
            print(f"  Warning: Could not load depth data: {e}")
    
    # Calculate Euler angles
    pitch, yaw, roll = rotmat_to_euler_xyz(R)
    
    # Generate gaze description
    gaze_desc = describe_gaze_direction(pitch, yaw, roll)
    
    # Create visualization with axes
    img_with_axes = draw_axes_on_image(img, R, T, axis_len=200)
    
    # Add Euler angle visualization
    img_with_viz = add_euler_visualization(img_with_axes, pitch, yaw, roll)
    
    # Add face mesh landmarks
    img_with_mesh = draw_face_mesh(img_with_viz)
    
    # Save result
    output_path = Path("outputs") / f"frame_{frame_num:05d}_with_axes.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img_with_mesh)
    print(f"  Pohrana: {output_path}")
    
    # Save depth visualization if available
    if depth_mm is not None:
        try:
            depth_viz = visualize_depth_map(depth_mm)
            depth_output = Path("outputs") / f"frame_{frame_num:05d}_depth.png"
            cv2.imwrite(str(depth_output), depth_viz)
            print(f"  Pohrana depth: {depth_output}")
        except Exception as e:
            print(f"  Warning: Could not save depth visualization: {e}")
    
    # Display image if requested
    if show_images:
        cv2.imshow(f"Frame {frame_num}: {gaze_desc}", img_with_mesh)
        if depth_mm is not None:
            depth_viz = visualize_depth_map(depth_mm)
            cv2.imshow(f"Frame {frame_num} - Depth", depth_viz)
        print(f"  Pritisnite bilo koju tipku za sljedeći frame...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    n_landmarks = len(get_face_landmarks(img))

    return {
        "Frame": frame_num,
        "Pitch (°)": f"{pitch:.2f}",
        "Yaw (°)": f"{yaw:.2f}",
        "Roll (°)": f"{roll:.2f}",
        "T (X, Y, Z)": f"({T[0]:.1f}, {T[1]:.1f}, {T[2]:.1f})",
        "Gaze Direction": gaze_desc,
        "Depth Range": depth_info,
        "Face Landmarks": n_landmarks
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
    
    print("\n" + "="*130)
    print("ZADATAK 1: Vizualna Interpretacija BIWI Head Pose Podataka sa HPE Utilities")
    print("Student: Krešimir Hartl (24)")
    print("="*130)
    
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
        
        print("\n" + "="*130)
        print("TABLICA REZULTATA")
        print("="*130)
        print(df.to_string(index=False))
        
        # Save table to CSV
        output_csv = Path("outputs/task1_results.csv")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nTabličnih rezultata pohranjena: {output_csv}")
    
    print("\n" + "="*130)
    print("KALIBRACIJA - DEBUG INFORMACIJE:")
    print("="*130)
    print("""
VIZUALIZACIJA SUSTAVA:
- Crvenni X: Kamera (0, 0, 0) - referentna točka
- Cyan kružnica: Glava - gdje T pokazuje
- Crvena os (X): Desna-lijeva
- Zelena os (Y): Gore-dolje  
- Plava os (Z): Naprijed-natrag

Ako je koordinatni sustav i dalje krivo pomaknut:
1. Provjerite je li T u pravim jedinicama (mm vs m)
2. Provjerite je li K matrica ispravna za vašu kameru
3. Provjerite je li R matrica u ispravnom redoslijedu (row-major vs col-major)
""")
    print("="*130 + "\n")


if __name__ == "__main__":
    main()
