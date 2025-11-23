import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load YOLOv11 pose model
model = YOLO("yolo11n-pose.pt")  

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Align depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Wait for frames and align them
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned depth and color frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        results = model(color_image, verbose=False)

        # Apply colormap to depth image for visualization
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03),
            cv2.COLORMAP_JET
        )

        # Draw keypoints from YOLO results 
        for r in results:
            keypoints = r.keypoints
            if keypoints is not None and keypoints.xy is not None:
                pts = keypoints.xy.cpu().numpy()
                for person in pts:
                    for x, y in person:
                        x, y = int(x), int(y)
                        if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
                            depth_value = aligned_depth_frame.get_distance(x, y)
                            cv2.circle(depth_colormap, (x, y), 4, (255, 255, 255), -1)
                            cv2.putText(depth_colormap,f"{depth_value:.2f}m",(x + 5, y - 5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 255, 255),1,)

                            cv2.circle(color_image, (x, y), 4, (255, 255, 255), -1)
                            cv2.putText(color_image,f"{depth_value:.2f}m",(x + 5, y - 5),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255, 255, 255),1,)

        #Stacking colour and depth images together
        stacked = np.hstack((color_image, depth_colormap))
        
        stacked = cv2.resize(stacked, (int(stacked.shape[1]*1.5), int(stacked.shape[0]*1.5)))
        # Show aligned depth with keypoints
        cv2.imshow("YOLOv11 Pose on RGB and Depth", stacked)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
