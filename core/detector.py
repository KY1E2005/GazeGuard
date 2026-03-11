import cv2
import mediapipe as mp
import numpy as np
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ObjectDetector:
    """Handles real-time object detection using MediaPipe Tasks API. 
        configured to detect mobile phones to prevent cheating or distractions."""

    def __init__(self, model_name='efficientdet_lite0.tflite'):
        # Load Model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_name)
        # Ensure the model file exists before proceeding
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        with open(model_path, 'rb') as f:
            model_content = f.read()

        # Configure the BaseOptions with the model buffer
        base_options = python.BaseOptions(model_asset_buffer=model_content)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5,
            #Restrict the AI to only look for phones
            category_allowlist=['cell phone', 'mobile phone']
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        
        # State Variables for Detection Smoothing
        self.phone_detected = False
        self.phone_box = None
        self.last_phone_time = 0 
        self.frame_count = 0
        self.SKIP_FRAMES = 15

    def detect(self, frame):
        """Analyzes a video frame to check for the presence of a mobile phone.
        Uses a frame-skipping mechanism and memory smoothing to reduce lag and flickering."""
        self.frame_count += 1
        
        # Only run the heavy object detection every 15 frames
        if self.frame_count % self.SKIP_FRAMES == 0:
            
            # Convert Frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Run Detection
            detection_result = self.detector.detect(mp_image)
            
            # Check for Phone
            found_now = False
            new_box = None
            for detection in detection_result.detections:
                for category in detection.categories:
                    # If a phone is found, grab its bounding box coordinates
                    if category.category_name in ['cell phone', 'mobile phone']:
                        found_now = True
                        bbox = detection.bounding_box
                        new_box = [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height]
                        break
                if found_now: break # Stop checking other detections if a phone is already confirmed
            
            # If a phone is seen, instantly update the state and record the exact time
            if found_now:
                self.phone_detected = True
                self.phone_box = new_box
                self.last_phone_time = time.time()
            else:
                if time.time() - self.last_phone_time > 0.9: # wait for 0.9 seconds before officially declaring it "gone".
                    self.phone_detected = False
                    self.phone_box = None
    
        return self.phone_detected, self.phone_box