import cv2
import mediapipe as mp

# --- 1. Inicijalizacija FaceMesh modela ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6
)

# --- 2. Učitavanje slike ---
image_path = "./data/raw/hpe/24/frame_00508_rgb.png"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError("Ne mogu učitati sliku!")

# Pretvori u RGB (MediaPipe zahtijeva RGB ulaz)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# --- 3. Izvršavanje FaceMesh modela ---
results = face_mesh.process(image_rgb)

# --- 4. Ako je pronađeno lice, nacrtaj svih 468 točaka ---
if results.multi_face_landmarks:
    for face in results.multi_face_landmarks:
        for lm in face.landmark:
            h, w, _ = image.shape
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

# --- 5. Prikaži sliku ---
cv2.imshow("MediaPipe FaceMesh — 468 keypointa", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
