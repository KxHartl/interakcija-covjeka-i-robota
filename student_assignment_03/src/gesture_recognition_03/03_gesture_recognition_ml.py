import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

print("\n" + "="*60)
print("REAL-TIME GESTURE RECOGNITION - ML MODEL")
print("="*60)

# Provjeri da li model datoteke postoje
model_path = 'gesture_model.pkl'
scaler_path = 'scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print(f"\n❌ GREŠKA: Model datoteke nisu pronaden!")
    print(f"   - {model_path}")
    print(f"   - {scaler_path}")
    print("\nTrebas prvo pokrenuti 02_train_gesture_model.py")
    exit(1)

# Ucitaj model i scaler
print(f"\n1. Ucitavanje modela...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"   ✓ Model ucitan")

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
print(f"   ✓ Scaler ucitan")

# Inicijalizacija MediaPipe
print(f"\n2. Inicijalizacija MediaPipe Hands...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
print(f"   ✓ MediaPipe Hands inicijalizirana")

# Inicijalizacija kamere
print(f"\n3. Inicijalizacija kamere...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
print(f"   ✓ Kamera je pokrenuta")
print(f"\n{'='*60}")
print("REAL-TIME PREDIKCIJA:")
print("Press ESC ili 'q' za izlaz")
print(f"{'='*60}\n")

prediction_buffer = []
buffer_size = 5

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Prikaži header
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.putText(frame, "Real-time Gesture Recognition (ML Model)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Geste: THUMBS_UP, THUMBS_DOWN",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Pretvori landmarks u feature vektor
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y])

                # Normalizacija
                features_scaled = scaler.transform([features])

                # Predikcija
                prediction = model.predict(features_scaled)[0]
                confidence_scores = model.predict_proba(features_scaled)[0]

                # Koristi buffer za gladjenje rezultata
                prediction_buffer.append(prediction)
                if len(prediction_buffer) > buffer_size:
                    prediction_buffer.pop(0)

                # Glađena predikcija (majority vote)
                smoothed_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                confidence = max(confidence_scores) * 100

                # Prikaži gestu
                gesture_names = ["THUMBS UP", "THUMBS DOWN"]
                gesture_name = gesture_names[int(smoothed_prediction)]
                gesture_color = (0, 255, 0) if smoothed_prediction == 0 else (0, 0, 255)

                # Crtaj landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=gesture_color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=gesture_color, thickness=2))

                # Prikaži label sa rezultatom
                wrist = hand_landmarks.landmark[0]
                px = int(wrist.x * w)
                py = int(wrist.y * h)

                cv2.rectangle(frame, (max(px-100, 0), max(py-50, 0)),
                            (min(px+100, w), min(py+30, h)), gesture_color, -1)
                cv2.putText(frame, f"{gesture_name}",
                           (max(px-90, 10), max(py-20, 30)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Confidence: {confidence:.1f}%",
                           (max(px-90, 10), max(py+5, 45)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        else:
            cv2.putText(frame, "No hand detected", (w//2-150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Gesture Recognition - ML Model (THUMBS UP/DOWN)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Zatvoren video feed")
    print(f"{'='*60}\n")
