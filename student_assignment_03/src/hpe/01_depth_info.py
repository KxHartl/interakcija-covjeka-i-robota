import numpy as np
import cv2
from pathlib import Path

# funkcija za dekodiranje podataka
def decode_biwi_depth(filepath: str) -> np.ndarray:
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

    # dodaj border pa resize na 640x480 (BIWI specifično)
    d_ = np.zeros((depth.shape[0] + 20, depth.shape[1] + 20), dtype=np.uint16)
    d_[20:, :640] = depth
    depth = cv2.resize(d_, (640, 480), interpolation=cv2.INTER_NEAREST)

    return depth

if __name__ == "__main__":
    # prilagodi putanju i frame_id
    root = Path("./data/raw/hpe/24")
    frame_id = 50

    depth_bin_path = root / f"frame_{frame_id:05d}_depth.bin"
    print(f"Čitam depth iz: {depth_bin_path}")

    depth_mm = decode_biwi_depth(str(depth_bin_path))      # uint16, milimetri 
    depth_m  = depth_mm.astype(np.float32) / 1000.0        # u metre

    h, w = depth_m.shape

    # osnovne info
    print(f"Dimenzije depth mape: {w} x {h}")
    print(f"Tip podataka: {depth_m.dtype}")

    valid = depth_m[depth_m > 0]

    if valid.size > 0:
        print(f"Min dubina (m): {valid.min():.3f}")
        print(f"Maks dubina (m): {valid.max():.3f}")
    else:
        print("Nema valjanih ( > 0 ) depth vrijednosti.")

    # primjer: dubina u centru slike
    cy, cx = h // 2, w // 2
    print(f"Depth u centru (px=({cx},{cy})) = {depth_m[cy, cx]:.3f} m")

        # ---------------- VIZUALIZACIJA ----------------
    # normaliziraj depth u [0, 255] za prikaz
    if valid.size > 0:
        depth_vis = cv2.normalize(depth_m, None, 0, 255, cv2.NORM_MINMAX)
        depth_vis = depth_vis.astype(np.uint8)

        # verzija u sivim tonovima
        gray = depth_vis.copy()

        cv2.imshow("Depth (gray)", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Nema što prikazati - depth je svugdje 0.")

