import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_TIP = 12
RING_TIP = 16
PINKY_TIP = 20

THUMB_PIP = 3
INDEX_PIP = 6
MIDDLE_PIP = 10
RING_PIP = 14
PINKY_PIP = 18

WRIST = 0

def is_finger_open(landmarks, tip_idx, pip_idx):
    """
    Provjerava je li prst otvoren (ispruzen) ili zatvoren (savijena).
    Koristi normaliziranu distancu od zapestja za robusnost.
    """
    wrist = landmarks[WRIST]
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    hand_scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    tip = landmarks[tip_idx]
    pip = landmarks[pip_idx]
    def _dist(a, b):
        return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    tip_w = _dist(tip, wrist) / hand_scale
    pip_w = _dist(pip, wrist) / hand_scale

    return tip_w > (pip_w + 0.12)

def distance_between_points(p1, p2):
    """
    Izracunava euklidsku distancu izmeu dvije tocke.
    """
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def detect_peace_gesture(landmarks):
    """
    PEACE gesta: Indeks i srednji prst su isprueni i separirani,
    palac i ostali prsti su savijena.
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    hand_scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    index_open = is_finger_open(landmarks, INDEX_TIP, INDEX_PIP)
    middle_open = is_finger_open(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    thumb_closed = not is_finger_open(landmarks, THUMB_TIP, THUMB_PIP)
    ring_closed = not is_finger_open(landmarks, RING_TIP, RING_PIP)
    pinky_closed = not is_finger_open(landmarks, PINKY_TIP, PINKY_PIP)

    separation = distance_between_points(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP]) / hand_scale
    separation_sufficient = separation > 0.15

    return (index_open and middle_open and thumb_closed and ring_closed and pinky_closed and separation_sufficient)

def detect_ok_gesture(landmarks):
    """
    OK gesta: Palac i kaziprst se dodiruju, ostali prsti su isprueni.
    """
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    hand_scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    middle_open = is_finger_open(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_open = is_finger_open(landmarks, RING_TIP, RING_PIP)
    pinky_open = is_finger_open(landmarks, PINKY_TIP, PINKY_PIP)

    thumb_index_distance = distance_between_points(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / hand_scale
    thumb_index_touching = thumb_index_distance < 0.12

    if thumb_index_touching and middle_open and ring_open and pinky_open:
        return True
    return False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for i, raw in enumerate(results.multi_hand_landmarks):
                landmarks = raw.landmark

                # Detektiranje geste
                if detect_peace_gesture(landmarks):
                    gesture = "PEACE"
                elif detect_ok_gesture(landmarks):
                    gesture = "OK"
                else:
                    gesture = "Hand"

                # Koristi MediaPipe-ovu klasifikaciju ruke - pouzdan je model
                hand_label = "Unknown"
                if results.multi_handedness and len(results.multi_handedness) > i:
                    try:
                        hand_label = results.multi_handedness[i].classification[0].label
                    except Exception:
                        hand_label = "Unknown"

                # Boja prema ruci: Right=plava, Left=crvena
                if hand_label == 'Right':
                    color = (255, 0, 0)  # Blue in BGR
                elif hand_label == 'Left':
                    color = (0, 0, 255)  # Red in BGR
                else:
                    color = (0, 255, 0)  # Green

                # Crtaj landmarke sa bojom
                mp_drawing.draw_landmarks(
                    frame,
                    raw,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2))

                # Prikazi label blizu zapestja
                wrist = landmarks[WRIST]
                px = int(wrist.x * w)
                py = int(wrist.y * h)
                label = f"{hand_label}: {gesture}"
                cv2.putText(frame, label, (max(px-50, 0), max(py-10, 20)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Gesture Recognition - PEACE / OK", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
