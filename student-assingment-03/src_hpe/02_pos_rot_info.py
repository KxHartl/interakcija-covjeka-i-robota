import numpy as np
import math
from pathlib import Path


def load_biwi_pose_txt(path: str):

    data = np.loadtxt(path)

    if data.shape != (4, 3):
        raise ValueError(f"Neočekivan oblik pose datoteke {path}: {data.shape}")

    R = data[:3, :]                 # 3x3 rotacijska matrica
    T_mm = data[3, :]               # translacija u mm
    T_m = T_mm.astype(np.float32) / 1000.0  # u metre

    return R.astype(np.float32), T_m


def rotmat_to_euler_xyz(R: np.ndarray):

    # pitch = rotacija oko X
    # yaw   = rotacija oko Y
    # roll  = rotacija oko Z

    assert R.shape == (3, 3)

    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(R[2, 1], R[2, 2])   # oko X
        yaw   = math.atan2(-R[2, 0], sy)       # oko Y
        roll  = math.atan2(R[1, 0], R[0, 0])   # oko Z
    else:
        # gimbal lock slučaj (rijetko)
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw   = math.atan2(-R[2, 0], sy)
        roll  = 0.0

    return np.degrees([pitch, yaw, roll])


if __name__ == "__main__":
    root = Path(r"C:\HRI\mediapipe_vjezba\data\raw\hpe\03")
    frame_id = 100

    pose_path = root / f"frame_{frame_id:05d}_pose.txt"
    print(f"Čitam pozu iz: {pose_path}")

    R, T_m = load_biwi_pose_txt(str(pose_path))

    print("\nRotacijska matrica R:")
    print(R)

    print("\nTranslacija T (m):")
    print(f"tx = {T_m[0]:.3f} m, ty = {T_m[1]:.3f} m, tz = {T_m[2]:.3f} m")

    pitch, yaw, roll = rotmat_to_euler_xyz(R)
    print("\nEuler kutovi (xyz):")
    print(f"Pitch (oko X) = {pitch:.2f}°")
    print(f"Yaw   (oko Y) = {yaw:.2f}°")
    print(f"Roll  (oko Z) = {roll:.2f}°")
