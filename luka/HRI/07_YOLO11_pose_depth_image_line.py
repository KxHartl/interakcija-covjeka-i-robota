import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import supervision as sv

class StoreKeypoints():
    def __init__(self,keypoint_detections,i):
        self.NOSE=keypoint_detections.xy[0][0]
        self.LEFT_EYE=keypoint_detections[0].xy[0][1]
        self.RIGHT_EYE=keypoint_detections[0].xy[0][2]
        self.LEFT_EAR=keypoint_detections[0].xy[0][3]
        self.RIGHT_EAR=keypoint_detections[0].xy[0][4]
        self.LEFT_SHOULDER=keypoint_detections[0].xy[0][5]
        self.RIGHT_SHOULDER=keypoint_detections[0].xy[0][6]
        self.LEFT_ELBOW=keypoint_detections[0].xy[0][7]
        self.RIGHT_ELBOW=keypoint_detections[0].xy[0][8]
        self.LEFT_WRIST=keypoint_detections[0].xy[0][9]
        self.RIGHT_WRIST=keypoint_detections[0].xy[0][10]
        self.LEFT_HIP=keypoint_detections[0].xy[0][11]
        self.RIGHT_HIP=keypoint_detections[0].xy[0][12]
        self.LEFT_KNEE=keypoint_detections[0].xy[0][13]
        self.RIGHT_KNEE=keypoint_detections[0].xy[0][14]
        self.LEFT_ANKLE=keypoint_detections[0].xy[0][15]
        self.RIGHT_ANKLE=keypoint_detections[0].xy[0][16]

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

        # Check if results are obtained
        if results and len(results) > 0:
            # Extract keypoints from the first result
            keypoints = sv.KeyPoints.from_ultralytics(results[0])
            for i in range(len(keypoints[0].xy)):
                
                # Check if any keypoints are detected
                if len(keypoints) > 0:
                    keypoint_names=StoreKeypoints(keypoint_detections=keypoints,i=i)

                    #Top of the chest keypoint (center between )
                    top_center_body_keypoint=((keypoint_names.LEFT_SHOULDER[0]+keypoint_names.RIGHT_SHOULDER[0])/2,(keypoint_names.LEFT_SHOULDER[1]+keypoint_names.RIGHT_SHOULDER[1])/2)
                    
                    #WORKING PHASE
                    
                    #Center between hips
                    bottom_center_body_keypoint=((keypoint_names.LEFT_HIP[0]+keypoint_names.RIGHT_HIP[0])/2,(keypoint_names.LEFT_HIP[1]+keypoint_names.RIGHT_HIP[1])/2)

                    x=np.linspace(top_center_body_keypoint[0], bottom_center_body_keypoint[0])

                    y=np.linspace(top_center_body_keypoint[1], bottom_center_body_keypoint[1])

                    points = list(zip(x.astype(int), y.astype(int)))

                    depth_values=[]
                    for point in points:
                        cv2.circle(depth_colormap, (point[0], point[1]), 2, (255,255,255), -1)
                        cv2.circle(color_image, (point[0], point[1]), 2, (255,255,255), -1)
                        try:
                            depth_values.append(aligned_depth_frame.get_distance(point[0], point[1]))
                        except:
                            pass
                    approx_depth = sum(depth_values)/len(depth_values)

                    cv2.putText(color_image, f"Distance: {approx_depth:.2f}m", (50,200), 2, 0.8, (255,255,255), 2)


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
