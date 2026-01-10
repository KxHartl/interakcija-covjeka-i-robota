import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path

# Use the standard mediapipe Hands solution (no external .task files required)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create the Hands object (will be used in the capture loop)
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# --- Konstante za landmark indekse ---
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
    Prst je otvoren ako je tip iznad PIP zgloba (manja y-koordinata).
    """
    # Use wrist-relative distances normalized by hand size to decide extension
    wrist = landmarks[WRIST]
    # hand bounding box scale
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


def classify_hand_by_landmarks(landmarks):
    """
    Heuristic classification of hand side from landmarks (person-centric).
    Uses the relative positions of index MCP (5) and pinky MCP (17) around the wrist (0).
    Returns 'Right' or 'Left'. This is independent from camera mirroring.
    """
    try:
        w = landmarks[WRIST]
        idx = landmarks[5]
        pky = landmarks[17]
        v1 = np.array([idx.x - w.x, idx.y - w.y])
        v2 = np.array([pky.x - w.x, pky.y - w.y])
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        # cross_z sign indicates orientation: positive -> index to pinky turns CCW
        if cross_z > 0:
            return 'Right'
        else:
            return 'Left'
    except Exception:
        return None

def detect_peace_gesture(landmarks):
    """
    PEACE gesta: Indeks i srednji prst su isprueni i separirani,
    palac i ostali prsti su savijena.
    
    Kriteriji:
    1. Indeks i srednji prst su OTVORENI (ispruzeni)
    2. Palac, prsten i mali prst su ZATVORENI (savijeni)
    3. Indeks i srednji prst imaju dovoljnu separaciju
    """
    # Compute normalized separation scale
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    hand_scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    index_open = is_finger_open(landmarks, INDEX_TIP, INDEX_PIP)
    middle_open = is_finger_open(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    thumb_closed = not is_finger_open(landmarks, THUMB_TIP, THUMB_PIP)
    ring_closed = not is_finger_open(landmarks, RING_TIP, RING_PIP)
    pinky_closed = not is_finger_open(landmarks, PINKY_TIP, PINKY_PIP)

    # Separacija indeksa i srednjeg prsta normalized
    separation = distance_between_points(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP]) / hand_scale
    separation_sufficient = separation > 0.15

    return (index_open and middle_open and thumb_closed and ring_closed and pinky_closed and separation_sufficient)

def detect_ok_gesture(landmarks):
    """
    OK gesta: Palac i kaziprst tvore krug (vrh se dodiruje ili je vrlo blizu),
    ostali prsti su isprueni.
    
    Kriteriji:
    1. Palac i kaziprst su ZATVORENI (vrh se dodiruje)
    2. Srednji, prsten i mali prst su OTVORENI (ispruzeni)
    3. Distanca izmeu vrha palca i vrha kaziprstao je vrlo mala (< 0.05)
    """
    # Use hand scale to normalize distances
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    hand_scale = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)

    middle_open = is_finger_open(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_open = is_finger_open(landmarks, RING_TIP, RING_PIP)
    pinky_open = is_finger_open(landmarks, PINKY_TIP, PINKY_PIP)

    # Thumb and index touching
    thumb_index_distance = distance_between_points(landmarks[THUMB_TIP], landmarks[INDEX_TIP]) / hand_scale
    thumb_index_touching = thumb_index_distance < 0.12

    # index and thumb may be considered "closed" when touching
    if thumb_index_touching and middle_open and ring_open and pinky_open:
        return True
    return False

# --- Kamera laptopa (OpenCV) ---
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

        detected_gesture = "Unknown"

        if results.multi_hand_landmarks:
            h, w, _ = frame.shape
            for i, raw in enumerate(results.multi_hand_landmarks):
                landmarks = raw.landmark

                if detect_peace_gesture(landmarks):
                    gesture = "PEACE"
                elif detect_ok_gesture(landmarks):
                    gesture = "OK"
                else:
                    gesture = "Hand"

                # Handedness label if available
                hand_label = None
                if results.multi_handedness and len(results.multi_handedness) > i:
                    try:
                        hand_label = results.multi_handedness[i].classification[0].label
                    except Exception:
                        hand_label = None

                # Choose color per hand using handedness when available: Right=blue, Left=red
                if hand_label is not None:
                    if hand_label.lower() == 'right':
                        color = (255, 0, 0)  # blue in BGR
                    elif hand_label.lower() == 'left':
                        color = (0, 0, 255)  # red in BGR
                    else:
                        color = (0, 255, 0)
                else:
                    color = (0, 255, 0) if i == 0 else (255, 128, 0)

                # Draw landmarks with custom color
                mp_drawing.draw_landmarks(
                    frame,
                    raw,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=color, thickness=2))

                # Display label near wrist
                wrist = landmarks[WRIST]
                px = int(wrist.x * w)
                py = int(wrist.y * h)
                label = f"{hand_label or ('Hand'+str(i+1))}: {gesture}"
                cv2.putText(frame, label, (max(px-10,0), max(py-10,20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Gesture Recognition - Custom Gestures (PEACE / OK)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC za izlaz
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
