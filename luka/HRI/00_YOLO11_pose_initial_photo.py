from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the YOLO11 pose model 
model = YOLO("yolo11n-pose.pt") 

# Run pose estimation on an image
results = model.predict(source='RealSense_RGB.png')  

# Show results with keypoints overlaid
result_img_bgr = results[0].plot()
result_img_rgb = result_img_bgr[..., ::-1]  # BGR to RGB conversion
plt.axis('off')
plt.imshow(result_img_rgb)
plt.show()
