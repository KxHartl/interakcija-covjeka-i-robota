import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import time

fps_start_time = time.time()
fps_frame_count = 0
fps = 0

pipeline=rs.pipeline()

config=rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

# Load the YOLO11 pose model
model = YOLO("yolo11s-pose.pt") 

try:
    while True:
        frames=pipeline.wait_for_frames()
        depth_frame=frames.get_depth_frame()
        color_frame=frames.get_color_frame()

        fps_frame_count += 1
        elapsed_time = time.time() - fps_start_time

        if elapsed_time > 1.0:  # Ažurira FPS svaku sekunu
            fps = fps_frame_count / elapsed_time
            fps_frame_count = 0
            fps_start_time = time.time()

        if not depth_frame or not color_frame:
            continue

        # Convert depth and color images to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        # Run YOLOv11 Pose model
        results = model(color_image, verbose=False)

        # Draw results on the frame
        annotated_frame = results[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (annotated_frame.shape[1]*2, annotated_frame.shape[0]*2))
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Show the output
        cv2.imshow("YOLOv11 Pose + RealSense", annotated_frame)
        


        # Exit on pressing the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


finally:
        pipeline.stop()

        cv2.destroyAllWindows()