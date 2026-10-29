# Human-robot interaction: motion classification, pose and gesture recognition

**Interakcija čovjeka i robota**

> Course exercises – Human-Robot Interaction (Interakcija čovjeka i robota), FSB, University of Zagreb, 2025/2026

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-pose%20%C2%B7%20face%20%C2%B7%20gestures-0097A7)
![YOLO11](https://img.shields.io/badge/Ultralytics-YOLO11%20pose-111F68)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-LDA-F7931E?logo=scikitlearn&logoColor=white)
![RealSense](https://img.shields.io/badge/camera-Intel%20RealSense-0071C5)

Assignments for the Human-Robot Interaction course. The first one classifies hand-guided motions
of a collaborative robot as fast or slow from recorded position and force. The others work with
people seen by a camera: detecting a keypoint in a live feed, estimating body and head pose, and
recognising hand gestures.

## What's inside

| Folder | What it does |
|---|---|
| `student_assignment_01` | Fast and slow motion classification. Tool position and force were recorded at 100 Hz while a Universal Robots cobot was guided by hand. The scripts remove idle segments, compute average speed and force per recording, train an LDA classifier and validate it on recordings from other groups. Report: [PDF](student_assignment_01/Kresimir_Hartl_Z1.pdf) and [Markdown](student_assignment_01/Izvjesce_Z1_Hartl.md) (Croatian). |
| `student_assignment_02` | Nose detection with YOLO11 pose on a laptop camera. The program draws a rectangle that turns green when the nose is inside it and red when it is outside. |
| `student_assignment_03` | Body pose, head pose and gesture recognition. MediaPipe keypoints are compared with the PennAction ground truth, joint angles and body tilt are computed, head orientation is checked against the BIWI head pose data and used to classify gaze direction, and new gestures are recognised both with hand-written rules and with a trained classifier. Report: [PDF](student_assignment_03/report/Kresimir_Hartl_Z2.pdf) (Croatian). |
| `skeletonizacija` | Exercise scripts for body keypoints with MediaPipe, from a single image to a RealSense feed, and a comparison with PennAction ground truth. |
| `luka/HRI` | YOLO11 pose example scripts, step by step from a photo to depth from a RealSense camera, with model weights. |

## How to run

Each assignment has its own requirements file.

```bash
# nose detection (assignment 2)
cd student_assignment_02/src
pip install -r requirements.txt
python nose_detection.py

# pose, head and gesture scripts (assignment 3)
cd student_assignment_03
pip install -r requirements.txt
python src/body_pose_estimation_01.py
```

More detail on assignment 2, including troubleshooting, is in
[`student_assignment_02/README.md`](student_assignment_02/README.md) (Croatian).
