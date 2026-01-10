import numpy as np
import cv2
from pathlib import Path

K = np.array([[575.816, 0.0, 320.0],
              [0.0, 575.816, 240.0],
              [0.0,   0.0,   1.0]], dtype=np.float32)

def load_biwi_pose_txt(path: str):
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


def project_point(P):
    p = K @ P
    u = p[0] / p[2]
    v = p[1] / p[2]
    return int(round(u)), int(round(v))


def draw_axes_perspective(img, R, T_mm, axis_len_mm=200.0):

    # ishodište (glava)
    P0 = T_mm

    # lokalne osi glave
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    # ako želiš da Z ide "iz nosa prema kameri", okreni znak
    #z_axis = -z_axis

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


if __name__ == "__main__":
    root = Path("./data/raw/hpe/24")
    frame_id = 350

    rgb_path  = root / f"frame_{frame_id:05d}_rgb.png"
    pose_path = root / f"frame_{frame_id:05d}_pose.txt"

    img = cv2.imread(str(rgb_path))
    R, T_mm = load_biwi_pose_txt(str(pose_path))

    img_axes = draw_axes_perspective(img.copy(), R, T_mm, axis_len_mm=200)

    cv2.imshow("BIWI Head Pose – 3D osi u perspektivi", img_axes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
