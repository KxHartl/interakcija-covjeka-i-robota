import mediapipe as mp

for lm in mp.solutions.pose.PoseLandmark:
    print(lm.value, lm.name)
