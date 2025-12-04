import cv2
import mediapipe as mp
import numpy as np

def calculate_angle(a, b, c):
    
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    ba = a - b
    bc = c - b

    # zaštita od nule
    if np.linalg.norm(ba) < 1e-6 or np.linalg.norm(bc) < 1e-6:
        return np.nan

    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # numerička stabilnost

    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)

# ---------- 1. Učitaj sliku ----------
putanja = "data/raw/1137/000001.jpg"
slika = cv2.imread(putanja)
if slika is None:
    raise FileNotFoundError(putanja)

h, w = slika.shape[:2]

# ---------- 2. MediaPipe Pose ----------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

with mp_pose.Pose(static_image_mode=True) as pose:
    rgb = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

if not results.pose_landmarks:
    print("Nema poze na slici.")
else:
    lm = results.pose_landmarks.landmark

    def to_px(lm_enum):
        p = lm[lm_enum.value]
        return [p.x * w, p.y * h]

    shoulder = to_px(mp_pose.PoseLandmark.LEFT_SHOULDER)
    elbow    = to_px(mp_pose.PoseLandmark.LEFT_ELBOW)
    wrist    = to_px(mp_pose.PoseLandmark.LEFT_WRIST)

    angle_elbow = calculate_angle(shoulder, elbow, wrist)
    print(f"Kut u lijevom laktu: {angle_elbow:.2f}°")

    # nacrtaj skeleton
    anotirana = slika.copy()
    mp_drawing.draw_landmarks(
        anotirana,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
    )

    cv2.imshow("Kut u laktu", anotirana)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
