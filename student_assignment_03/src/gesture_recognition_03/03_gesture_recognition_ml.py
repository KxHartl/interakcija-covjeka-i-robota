import cv2
import mediapipe as mp
import pickle
import numpy as np
import os

print("\n" + "="*60)
print("REAL-TIME GESTURE RECOGNITION - ML MODEL (IMPROVED)")
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
print("REAL-TIME PREDIKCIJA (sa validacijom geste):")
print("Press ESC ili 'q' za izlaz")
print(f"{'='*60}\n")

# ========== VALIDACIJSKI PRAGOVI ==========
MIN_CONFIDENCE_THRESHOLD = 0.75  # Minimum 75% sigurnosti za predikciju
MIN_CLASS_PROBABILITY_GAP = 0.15  # Razlika između top 2 klase mora biti >15%
CONFIDENCE_BUFFER_SIZE = 8       # Trebaj 8 uzastopnih frameova prije prikaza
GEST_VALIDATION_THRESHOLD = 0.85  # Minimum 85% za validaciju geste

# ========== HELPER FUNKCIJA ZA VALIDACIJU GESTE ==========
def validate_gesture(hand_landmarks):
    """
    Validira da li ruka ima pravilan oblik za THUMBS UP ili THUMBS DOWN.
    Vraća True ako je gesta validna.
    """
    try:
        # Ključne točke
        wrist = np.array([hand_landmarks[0].x, hand_landmarks[0].y])
        thumb_tip = np.array([hand_landmarks[4].x, hand_landmarks[4].y])
        index_mcp = np.array([hand_landmarks[5].x, hand_landmarks[5].y])
        middle_mcp = np.array([hand_landmarks[9].x, hand_landmarks[9].y])
        
        # Provjera 1: Palac mora biti jasno vidljiv
        thumb_distance = np.linalg.norm(thumb_tip - wrist)
        if thumb_distance < 0.05:  # Palac je previše blizu zapešća
            return False
        
        # Provjera 2: Ostatak prstiju trebao bi biti grupirani
        # Izračunaj bounding box od preostalih prstiju
        other_landmarks = [hand_landmarks[i] for i in range(5, 21)]
        finger_x = [lm.x for lm in other_landmarks]
        finger_y = [lm.y for lm in other_landmarks]
        
        finger_width = max(finger_x) - min(finger_x)
        finger_height = max(finger_y) - min(finger_y)
        
        # Prstiji trebali bi biti relativno kompaktni
        if finger_width > 0.25 or finger_height > 0.3:
            return False
        
        # Provjera 3: Palac i ostatak prstiju trebali bi biti odvojeni
        # Distanca od palca do indeksa trebala bi biti relativno velika
        index_tip = np.array([hand_landmarks[8].x, hand_landmarks[8].y])
        palac_index_distance = np.linalg.norm(thumb_tip - index_tip)
        
        if palac_index_distance < 0.08:  # Preriblizu su
            return False
        
        return True
        
    except Exception as e:
        print(f"Validacija greška: {e}")
        return False


def check_confidence_quality(confidence_scores):
    """
    Provjeri je li model dovoljno siguran u svoju predikciju.
    """
    sorted_scores = np.sort(confidence_scores)[::-1]
    max_confidence = sorted_scores[0]
    second_max_confidence = sorted_scores[1] if len(sorted_scores) > 1 else 0
    
    # Razlika između top 2 klase
    confidence_gap = max_confidence - second_max_confidence
    
    return max_confidence >= MIN_CONFIDENCE_THRESHOLD and confidence_gap >= MIN_CLASS_PROBABILITY_GAP


prediction_buffer = []
confidence_buffer = []
valid_gesture_counter = 0
display_gesture = None
display_confidence = 0

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
        cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
        cv2.putText(frame, "Real-time Gesture Recognition (ML Model + Validation)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Geste: THUMBS_UP, THUMBS_DOWN",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        cv2.putText(frame, f"Confidence threshold: {MIN_CONFIDENCE_THRESHOLD*100:.0f}% | Gap: {MIN_CLASS_PROBABILITY_GAP*100:.0f}%",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        gesture_detected = False

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # KORAK 1: Validacija geste - Provjeri oblik ruke
                is_valid = validate_gesture(hand_landmarks.landmark)
                
                # KORAK 2: Pretvori landmarks u feature vektor
                features = []
                for lm in hand_landmarks.landmark:
                    features.extend([lm.x, lm.y])

                # KORAK 3: Normalizacija
                features_scaled = scaler.transform([features])

                # KORAK 4: Predikcija
                prediction = model.predict(features_scaled)[0]
                confidence_scores = model.predict_proba(features_scaled)[0]
                max_confidence = max(confidence_scores)

                # KORAK 5: Provjera kvalitete pouzdanosti
                is_confident = check_confidence_quality(confidence_scores)

                # KORAK 6: Dodaj u buffer SAMO ako je gesta validna I model je siguran
                if is_valid and is_confident:
                    prediction_buffer.append(prediction)
                    confidence_buffer.append(max_confidence)
                    valid_gesture_counter += 1
                    
                    if len(prediction_buffer) > CONFIDENCE_BUFFER_SIZE:
                        prediction_buffer.pop(0)
                        confidence_buffer.pop(0)
                else:
                    # Resetuj buffer ako gesta nije validna
                    prediction_buffer.clear()
                    confidence_buffer.clear()
                    valid_gesture_counter = 0

                # KORAK 7: Prikaži gestu samo nakon dovoljno validnih frameova
                if len(prediction_buffer) >= CONFIDENCE_BUFFER_SIZE * 0.75:  # 75% buffer-a
                    smoothed_prediction = max(set(prediction_buffer), key=prediction_buffer.count)
                    avg_confidence = np.mean(confidence_buffer) * 100
                    
                    gesture_names = ["THUMBS UP", "THUMBS DOWN"]
                    display_gesture = gesture_names[int(smoothed_prediction)]
                    display_confidence = avg_confidence
                    gesture_detected = True
                    gesture_color = (0, 255, 0) if smoothed_prediction == 0 else (0, 0, 255)
                else:
                    display_gesture = None
                    gesture_detected = False

                # Crtaj landmarks
                if display_gesture:  # Samo ako je gesta validna
                    gesture_color = (0, 255, 0) if display_gesture == "THUMBS UP" else (0, 0, 255)
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

                    cv2.rectangle(frame, (max(px-110, 0), max(py-60, 0)),
                                (min(px+110, w), min(py+40, h)), gesture_color, -1)
                    cv2.putText(frame, f"{display_gesture}",
                               (max(px-100, 10), max(py-30, 30)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, f"Confidence: {display_confidence:.1f}%",
                               (max(px-100, 10), max(py+10, 50)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"Frames: {len(prediction_buffer)}/{CONFIDENCE_BUFFER_SIZE}",
                               (max(px-100, 10), max(py+30, 70)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                else:
                    # Crtaj samo landmarks u sivoj boji ako gesta nije validna
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1, circle_radius=1),
                        mp_drawing.DrawingSpec(color=(100, 100, 100), thickness=1))
                    
                    # Prikaži status
                    wrist = hand_landmarks.landmark[0]
                    px = int(wrist.x * w)
                    py = int(wrist.y * h)
                    cv2.putText(frame, "Waiting for valid gesture...",
                               (max(px-120, 10), max(py-20, 30)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        else:
            cv2.putText(frame, "No hand detected", (w//2-150, h//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Gesture Recognition - ML Model (THUMBS UP/DOWN) with Validation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Zatvoren video feed")
    print(f"{'='*60}\n")
