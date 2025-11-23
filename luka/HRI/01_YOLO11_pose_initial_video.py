from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO11 pose model
model = YOLO("yolo11s-pose.pt") 

# Run pose estimation on an image
results = model.predict(source='Template01.mov', show = True) 