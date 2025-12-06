#!/usr/bin/env python3
"""
Nose Detection and Classification Program
Detects student's nose position from camera and displays a rectangle that changes color
based on whether the nose is inside or outside the rectangle.

Author: Student Assignment 02
Date: 2025

Based on examples from /luka/HRI/ directory using YOLO11 pose detection
"""

import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import sys


class NoseDetector:
    """
    Class for detecting nose position using YOLO11 pose detection.
    """
    
    def __init__(self, model_path="yolo11s-pose.pt", rectangle_width=150, rectangle_height=200, confidence_threshold=0.5):
        """
        Initialize the nose detector.
        
        Args:
            model_path (str): Path to YOLO11 pose model
            rectangle_width (int): Width of the detection rectangle
            rectangle_height (int): Height of the detection rectangle
            confidence_threshold (float): Minimum confidence for detection
        """
        try:
            # Load YOLO11 pose model
            self.model = YOLO(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
        
        # Initialize video capture from webcam
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Cannot access camera")
            sys.exit(1)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get frame dimensions
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Rectangle parameters
        self.rectangle_width = rectangle_width
        self.rectangle_height = rectangle_height
        
        # Center rectangle position
        self.rect_center_x = self.frame_width // 2
        self.rect_center_y = self.frame_height // 2
        
        # YOLO11 keypoint indices
        self.NOSE_INDEX = 0  # Nose is at index 0 in YOLO11 pose
        
        # Colors in BGR format
        self.COLOR_GREEN = (0, 255, 0)    # Nose inside rectangle
        self.COLOR_RED = (0, 0, 255)      # Nose outside rectangle
        self.COLOR_WHITE = (255, 255, 255)
        self.COLOR_YELLOW = (0, 255, 255)
        self.COLOR_BLUE = (255, 0, 0)
        
        # Confidence threshold
        self.confidence_threshold = confidence_threshold
    
    def detect_nose_position(self, frame):
        """
        Detect nose position in the frame using YOLO11.
        
        Args:
            frame (np.ndarray): Input frame from camera
            
        Returns:
            tuple: (nose_x, nose_y, confidence) or (None, None, None) if not detected
        """
        # Run YOLO11 inference
        results = self.model(frame)
        
        if results and len(results) > 0:
            result = results[0]
            
            # Check if keypoints were detected
            if result.keypoints is not None and len(result.keypoints) > 0:
                # Extract keypoints using supervision library
                keypoints = sv.KeyPoints.from_ultralytics(result)
                
                if len(keypoints.xy) > 0:
                    # Get nose coordinates (index 0)
                    nose_coords = keypoints.xy[0][self.NOSE_INDEX]
                    nose_x = int(nose_coords[0])
                    nose_y = int(nose_coords[1])
                    
                    # Get confidence score
                    if keypoints.confidence is not None and len(keypoints.confidence) > 0:
                        confidence = keypoints.confidence[0][self.NOSE_INDEX]
                    else:
                        confidence = 1.0
                    
                    # Check if confidence is above threshold
                    if confidence >= self.confidence_threshold:
                        return nose_x, nose_y, confidence
        
        return None, None, None
    
    def is_point_in_rectangle(self, point_x, point_y, rect_x, rect_y, rect_w, rect_h):
        """
        Check if a point is inside a rectangle.
        
        Args:
            point_x (int): X coordinate of the point
            point_y (int): Y coordinate of the point
            rect_x (int): X coordinate of rectangle center
            rect_y (int): Y coordinate of rectangle center
            rect_w (int): Width of rectangle
            rect_h (int): Height of rectangle
            
        Returns:
            bool: True if point is inside rectangle, False otherwise
        """
        left = rect_x - rect_w // 2
        right = rect_x + rect_w // 2
        top = rect_y - rect_h // 2
        bottom = rect_y + rect_h // 2
        
        return left <= point_x <= right and top <= point_y <= bottom
    
    def draw_rectangle_and_nose(self, frame, nose_x, nose_y, is_inside):
        """
        Draw rectangle and nose point on the frame.
        
        Args:
            frame (np.ndarray): Frame to draw on
            nose_x (int): X coordinate of nose
            nose_y (int): Y coordinate of nose
            is_inside (bool): Whether nose is inside rectangle
            
        Returns:
            np.ndarray: Frame with drawings
        """
        # Determine rectangle color based on nose position
        rect_color = self.COLOR_GREEN if is_inside else self.COLOR_RED
        
        # Calculate rectangle boundaries
        left = self.rect_center_x - self.rectangle_width // 2
        top = self.rect_center_y - self.rectangle_height // 2
        right = self.rect_center_x + self.rectangle_width // 2
        bottom = self.rect_center_y + self.rectangle_height // 2
        
        # Draw rectangle with thick border
        cv2.rectangle(frame, (left, top), (right, bottom), rect_color, 3)
        
        # Draw center cross for reference
        cross_size = 20
        cv2.line(frame, 
                (self.rect_center_x - cross_size, self.rect_center_y),
                (self.rect_center_x + cross_size, self.rect_center_y),
                self.COLOR_BLUE, 2)
        cv2.line(frame,
                (self.rect_center_x, self.rect_center_y - cross_size),
                (self.rect_center_x, self.rect_center_y + cross_size),
                self.COLOR_BLUE, 2)
        
        # Draw nose point if detected
        if nose_x is not None and nose_y is not None:
            cv2.circle(frame, (nose_x, nose_y), 8, self.COLOR_YELLOW, -1)
            cv2.circle(frame, (nose_x, nose_y), 8, self.COLOR_WHITE, 2)
        
        return frame
    
    def draw_info_text(self, frame, nose_x, nose_y, is_inside, confidence):
        """
        Draw informational text on the frame.
        
        Args:
            frame (np.ndarray): Frame to draw on
            nose_x (int): X coordinate of nose
            nose_y (int): Y coordinate of nose
            is_inside (bool): Whether nose is inside rectangle
            confidence (float): Detection confidence
            
        Returns:
            np.ndarray: Frame with text
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        line_height = 30
        
        y_position = 30
        
        # Display status
        if nose_x is not None and nose_y is not None:
            status = "INSIDE" if is_inside else "OUTSIDE"
            status_color = self.COLOR_GREEN if is_inside else self.COLOR_RED
            
            cv2.putText(frame, f"Status: {status}", (10, y_position), 
                       font, font_scale, status_color, thickness)
            y_position += line_height
            
            cv2.putText(frame, f"Nose Position: ({nose_x}, {nose_y})", (10, y_position),
                       font, font_scale, self.COLOR_WHITE, thickness)
            y_position += line_height
            
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, y_position),
                       font, font_scale, self.COLOR_WHITE, thickness)
        else:
            cv2.putText(frame, "No face detected", (10, y_position),
                       font, font_scale, self.COLOR_WHITE, thickness)
        
        # Display instructions at bottom
        cv2.putText(frame, "Press 'q' to quit", (10, self.frame_height - 20),
                   font, font_scale, self.COLOR_WHITE, thickness)
        
        return frame
    
    def run(self):
        """
        Main loop for nose detection and visualization.
        """
        print("Starting nose detection...")
        print(f"Camera resolution: {self.frame_width}x{self.frame_height}")
        print(f"Rectangle size: {self.rectangle_width}x{self.rectangle_height}")
        print(f"Confidence threshold: {self.confidence_threshold}")
        print("Press 'q' to quit the application")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Error: Cannot read from camera")
                    break
                
                # Flip frame for selfie view
                frame = cv2.flip(frame, 1)
                
                frame_count += 1
                
                # Detect nose position
                nose_x, nose_y, confidence = self.detect_nose_position(frame)
                
                # Check if nose is inside rectangle
                is_inside = False
                if nose_x is not None and nose_y is not None:
                    is_inside = self.is_point_in_rectangle(
                        nose_x, nose_y,
                        self.rect_center_x, self.rect_center_y,
                        self.rectangle_width, self.rectangle_height
                    )
                
                # Draw rectangle and nose
                frame = self.draw_rectangle_and_nose(frame, nose_x, nose_y, is_inside)
                
                # Draw information text
                frame = self.draw_info_text(frame, nose_x, nose_y, is_inside, confidence if confidence else 0)
                
                # Display frame
                cv2.imshow('Nose Detection - Press q to quit', frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Application closed by user")
                    break
        
        except KeyboardInterrupt:
            print("Application interrupted by user")
        
        finally:
            # Cleanup
            print(f"Total frames processed: {frame_count}")
            self.cap.release()
            cv2.destroyAllWindows()
            print("Nose detection application stopped")


def main():
    """
    Main function to start the nose detection application.
    """
    # Try to use the model from /luka/ directory, otherwise use default
    import os
    
    model_path = "yolo11s-pose.pt"
    
    # Check if model exists in current directory
    if not os.path.exists(model_path):
        # Try to find it in parent directories
        parent_dir = os.path.dirname(os.path.abspath(__file__))
        for _ in range(3):  # Check up to 3 parent levels
            model_path_candidate = os.path.join(parent_dir, "luka", "HRI", "yolo11s-pose.pt")
            if os.path.exists(model_path_candidate):
                model_path = model_path_candidate
                break
            parent_dir = os.path.dirname(parent_dir)
    
    # Create nose detector instance
    detector = NoseDetector(
        model_path=model_path,
        rectangle_width=200,
        rectangle_height=250,
        confidence_threshold=0.5
    )
    
    # Run the detection
    detector.run()


if __name__ == "__main__":
    main()
