import cv2
import mediapipe as mp

putanja = 'data/raw/1137/000001.jpg'
slika = cv2.imread(putanja)

mp_pose = mp.solutions.pose

with mp_pose.Pose(static_image_mode=True) as pose:
    rgb = cv2.cvtColor(slika, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

if results.pose_landmarks:
    lm = results.pose_landmarks.landmark
    nose = lm[mp_pose.PoseLandmark.NOSE.value]

    h, w = slika.shape[:2]

    x_px = int(nose.x * w)
    y_px = int(nose.y * h)

    print("Normalizirano:", nose.x, nose.y)
    print("U pikselima:", x_px, y_px)
else:
    print("Nema poze.")
