import cv2
import mediapipe as mp
from pathlib import Path

"""
SET 24
frames: 50, 150, 250, 350, 420, 508
"""

# --- 1. Inicijalizacija FaceMesh modela ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# --- 2. Učitavanje slike ---
root = Path("./data/raw/hpe/24")
frame_id = 508

image_path = root / f"frame_{frame_id:05d}_rgb.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Ne mogu učitati sliku!")

# Pretvori u RGB (MediaPipe zahtijeva RGB ulaz)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- 3. Izvršavanje FaceMesh modela ---
results = face_mesh.process(image_rgb)

# --- 4. Ako je pronađeno lice, nacrtaj svih 468 točaka i prikaži broj ---
landmarks = []
if results.multi_face_landmarks:
    h, w, _ = image.shape
    for face in results.multi_face_landmarks:
        for lm in face.landmark:
            x = int(lm.x * w)
            y = int(lm.y * h)
            landmarks.append((x, y))

    # Draw landmarks
    for (x, y) in landmarks:
        cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # Count and display
    n_landmarks = len(landmarks)
    print(f"FaceMesh: pronađeno {n_landmarks} landmarka")
    cv2.putText(image, f'Landmarks: {n_landmarks}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

# --- 5. Prikaži sliku ---
cv2.imshow("MediaPipe FaceMesh", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
