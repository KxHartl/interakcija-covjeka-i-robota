import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

# --- MediaPipe Hands setup ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Putanja do hand landmarker modela
root = Path("./data/raw/gr")
model_path = root.parent / "hand_landmarker.task"

base_options = BaseOptions(model_asset_path=str(model_path))
options = HandLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.IMAGE
)

landmarker = HandLandmarker.create_from_options(options)

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
    if landmarks[tip_idx].y < landmarks[pip_idx].y:
        return True
    return False

def distance_between_points(p1, p2):
    """
    Izracunava euklidsku distancu izmeu dvije tocke.
    """
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def detect_peace_gesture(landmarks):
    """
    PEACE gesta: Indeks i srednji prst su isprueni i separirani,
    palac i ostali prsti su savijena.
    
    Kriteriji:
    1. Indeks i srednji prst su OTVORENI (ispruzeni)
    2. Palac, prsten i mali prst su ZATVORENI (savijeni)
    3. Indeks i srednji prst imaju dovoljnu separaciju
    """
    index_open = is_finger_open(landmarks, INDEX_TIP, INDEX_PIP)
    middle_open = is_finger_open(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    thumb_closed = not is_finger_open(landmarks, THUMB_TIP, THUMB_PIP)
    ring_closed = not is_finger_open(landmarks, RING_TIP, RING_PIP)
    pinky_closed = not is_finger_open(landmarks, PINKY_TIP, PINKY_PIP)
    
    # Separacija indeksa i srednjeg prsta
    separation = distance_between_points(landmarks[INDEX_TIP], landmarks[MIDDLE_TIP])
    separation_sufficient = separation > 0.05
    
    if (index_open and middle_open and thumb_closed and ring_closed and 
        pinky_closed and separation_sufficient):
        return True
    return False

def detect_ok_gesture(landmarks):
    """
    OK gesta: Palac i kaziprst tvore krug (vrh se dodiruje ili je vrlo blizu),
    ostali prsti su isprueni.
    
    Kriteriji:
    1. Palac i kaziprst su ZATVORENI (vrh se dodiruje)
    2. Srednji, prsten i mali prst su OTVORENI (ispruzeni)
    3. Distanca izmeu vrha palca i vrha kaziprstao je vrlo mala (< 0.05)
    """
    thumb_closed = not is_finger_open(landmarks, THUMB_TIP, THUMB_PIP)
    index_closed = not is_finger_open(landmarks, INDEX_TIP, INDEX_PIP)
    middle_open = is_finger_open(landmarks, MIDDLE_TIP, MIDDLE_PIP)
    ring_open = is_finger_open(landmarks, RING_TIP, RING_PIP)
    pinky_open = is_finger_open(landmarks, PINKY_TIP, PINKY_PIP)
    
    # Distanca izmeu vrha palca i vrha kaziprstoa - trebala bi biti mala
    thumb_index_distance = distance_between_points(landmarks[THUMB_TIP], landmarks[INDEX_TIP])
    thumb_index_touching = thumb_index_distance < 0.05
    
    if (thumb_closed and index_closed and middle_open and ring_open and 
        pinky_open and thumb_index_touching):
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
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = landmarker.detect(mp_image)

        detected_gesture = "Unknown"
        
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            
            if detect_peace_gesture(landmarks):
                detected_gesture = "PEACE"
            elif detect_ok_gesture(landmarks):
                detected_gesture = "OK"
            else:
                detected_gesture = "Hand detected"
            
            # Prikaz rezultata na slici
            cv2.putText(frame, f"Gesta: {detected_gesture}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        cv2.imshow("Gesture Recognition - Custom Gestures (PEACE / OK)", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC za izlaz
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
