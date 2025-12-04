import cv2
import mediapipe as mp

# Učitavanje slike
putanja = 'data/raw/1137/000001.jpg'
slika = cv2.imread(putanja)

if slika is None:
    raise FileNotFoundError(f"Ne mogu učitati sliku: {putanja}")

# MediaPipe moduli
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Konverzija u RGB (MediaPipe radi u RGB)
slika_rgb = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)

# Detekcija poze
with mp_pose.Pose(static_image_mode=True) as pose:
    results = pose.process(slika_rgb)

# Crtanje rezultata
if not results.pose_landmarks:
    print("Nema detektirane poze na slici.")
else:
    anotirana = slika.copy()
    mp_drawing.draw_landmarks(
        anotirana,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
    )

    cv2.imshow('Pose detection', anotirana)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
