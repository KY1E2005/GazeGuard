import cv2
import mediapipe as mp
import numpy as np
import os
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ObjectDetector:
    def __init__(self, model_name='efficientdet_lite0.tflite'):
        # 1. Load Model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_name)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at: {model_path}")

        with open(model_path, 'rb') as f:
            model_content = f.read()

        base_options = python.BaseOptions(model_asset_buffer=model_content)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5, 
            category_allowlist=['cell phone', 'mobile phone'] 
        )
        self.detector = vision.ObjectDetector.create_from_options(options)
        
        # State
        self.phone_detected = False
        self.last_phone_time = 0 
        self.frame_count = 0
        self.SKIP_FRAMES = 15

    def detect(self, frame):
        self.frame_count += 1
        if self.frame_count % self.SKIP_FRAMES == 0:
            
            # 1. Convert Frame
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # 2. Run Detection
            detection_result = self.detector.detect(mp_image)
            
            # 3. Check for Phone
            found_now = False
            for detection in detection_result.detections:
                for category in detection.categories:
                    if category.category_name in ['cell phone', 'mobile phone']:
                        found_now = True
                        break
            
            # 4. Logic with Memory (Cooldown)
            if found_now:
                self.phone_detected = True
                self.last_phone_time = time.time()
            else:
                # Keep detecting for 2.0 seconds after it disappears
                if time.time() - self.last_phone_time > 2.0:
                    self.phone_detected = False

        return self.phone_detected