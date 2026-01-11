import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

collected_data = []
current_gesture = None
frame_count = 0
target_frames = 50

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("\n" + "="*60)
print("GESTURE DATA COLLECTION - THUMBS UP vs THUMBS DOWN")
print("="*60)
print("\nInstrukcije:")
print("  'u' - Pocni prikupljati THUMBS UP podatke")
print("  'd' - Pocni prikupljati THUMBS DOWN podatke")
print("  'n' - Zaustavi prikupljanje (next gesture)")
print("  'q' - Spremi i zatvori")
print("\nTrebam 15 frameova za svaku gestu.")
print("="*60 + "\n")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # Prikaži instrukcije
        cv2.rectangle(frame, (10, 10), (w-10, 100), (0, 0, 0), -1)
        cv2.putText(frame, "GESTURE DATA COLLECTION", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if current_gesture is None:
            cv2.putText(frame, "Press 'u' for THUMBS_UP, 'd' for THUMBS_DOWN, 'q' quit",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        elif current_gesture == "thumbs_up":
            cv2.putText(frame, f"THUMBS UP - Recording: {frame_count}/{target_frames} (Press 'n' next, 'q' quit)",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        elif current_gesture == "thumbs_down":
            cv2.putText(frame, f"THUMBS DOWN - Recording: {frame_count}/{target_frames} (Press 'n' next, 'q' quit)",
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

        # Prikaži broj prikupljenih podataka po gestama
        thumbs_up_count = sum(1 for x in collected_data if x[-1] == 0)
        thumbs_down_count = sum(1 for x in collected_data if x[-1] == 1)
        cv2.putText(frame, f"Total: THUMBS_UP={thumbs_up_count}, THUMBS_DOWN={thumbs_down_count}",
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

        # Prikupljanje podataka
        if results.multi_hand_landmarks and current_gesture is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # Pretvori landmarks u row podataka
                row = []
                for lm in hand_landmarks.landmark:
                    row.extend([lm.x, lm.y])

                # Dodaj labelu (0=thumbs_up, 1=thumbs_down)
                label = 0 if current_gesture == "thumbs_up" else 1
                row.append(label)

                collected_data.append(row)
                frame_count += 1

                # Crtaj landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2))

                # Ako je prikupljeno dovoljno frameova
                if frame_count >= target_frames:
                    print(f"✓ Prikupljeno {target_frames} frameova za {current_gesture}")
                    current_gesture = None
                    frame_count = 0

        cv2.imshow("Gesture Data Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('u'):
            current_gesture = "thumbs_up"
            frame_count = 0
            print(f"\nPoceti prikupljanje THUMBS_UP...")
        elif key == ord('d'):
            current_gesture = "thumbs_down"
            frame_count = 0
            print(f"Poceti prikupljanje THUMBS_DOWN...")
        elif key == ord('n'):
            if current_gesture is not None:
                print(f"✓ Zaustavljeno prikupljanje {current_gesture} ({frame_count} frameova)")
            current_gesture = None
            frame_count = 0
        elif key == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()

# Spremi u CSV datoteku
if collected_data:
    # Kreiraj header
    header = []
    for i in range(21):
        header.extend([f'lm_{i}_x', f'lm_{i}_y'])
    header.append('label')

    # Spremi CSV
    csv_path = './outputs/gesture_data.csv'
    os.makedirs('.', exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(collected_data)

    print(f"\n{'='*60}")
    print(f"✓ Prikupljeni podaci: {len(collected_data)} frameova")
    print(f"  - THUMBS_UP: {sum(1 for x in collected_data if x[-1] == 0)}")
    print(f"  - THUMBS_DOWN: {sum(1 for x in collected_data if x[-1] == 1)}")
    print(f"✓ Spremi u: {csv_path}")
    print(f"{'='*60}")
else:
    print("\nNisu prikupljeni podaci!")
