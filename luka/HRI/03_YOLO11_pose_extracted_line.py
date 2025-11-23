import cv2
from ultralytics import YOLO
import supervision as sv
import math

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

# Load the YOLO model
model = YOLO("yolo11s-pose.pt")  # pretrained YOLOv8s-pose model

cap = cv2.VideoCapture('Template01.mov')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Run inference on the frame
    results = model(frame)  # returns a list of Results objects
    
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
                
                
                #Center between hips
                bottom_center_body_keypoint=((keypoint_names.LEFT_HIP[0]+keypoint_names.RIGHT_HIP[0])/2,(keypoint_names.LEFT_HIP[1]+keypoint_names.RIGHT_HIP[1])/2)

                if top_center_body_keypoint != (0.0,0.0) and bottom_center_body_keypoint != (0.0,0.0) and keypoint_names.LEFT_SHOULDER[0] != 0.0 and keypoint_names.LEFT_SHOULDER[1] != 0.0 and keypoint_names.RIGHT_SHOULDER[0] != 0.0 and keypoint_names.RIGHT_SHOULDER[1] != 0.0 and keypoint_names.LEFT_HIP[0] != 0.0 and keypoint_names.LEFT_HIP[1] != 0.0 and keypoint_names.RIGHT_HIP[0] != 0.0 and keypoint_names.RIGHT_HIP[1] != 0.0:
                    
                    cv2.circle(frame, (int(top_center_body_keypoint[0]), int(top_center_body_keypoint[1])), 8,(255,0,0),-1)
                    cv2.circle(frame, (int(bottom_center_body_keypoint[0]), int(bottom_center_body_keypoint[1])), 8,(255,0,0),-1)
                    cv2.line(frame,(int(top_center_body_keypoint[0]),int(top_center_body_keypoint[1])),(int(bottom_center_body_keypoint[0]), int(bottom_center_body_keypoint[1])),(255,255,255),8,4)

                    #Whatch out: if the point is not visible in the frame, the system by default sets the point values to 0
                elif keypoint_names.LEFT_SHOULDER[0] != 0.0 and keypoint_names.LEFT_SHOULDER[1] != 0.0 and keypoint_names.RIGHT_SHOULDER[0] != 0.0 and keypoint_names.RIGHT_SHOULDER[1] != 0.0:
                    cv2.circle(frame, (int(top_center_body_keypoint[0]), int(top_center_body_keypoint[1])), 6,(255,0,0),-1)

            
        else:
            print("No keypoints detected in the current frame.")
    else:
        print("No results obtained from the model.")
    #Resize the image
    frame = cv2.resize(frame, (int(frame.shape[1]/4), int(frame.shape[0]/4))) 

    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
