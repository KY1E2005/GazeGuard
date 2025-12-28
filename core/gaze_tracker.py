#handle all the interaction with MediaPipe

import cv2
import mediapipe as mp

class GazeTracker:
    def __init__(self):
        print("Loading MediaPipe FaceMesh...")
        self.mp_face_mesh = mp.solutions.face_mesh
        
        # Initialize FaceMesh with refined_landmarks=True
        # gives us the IRIS landmarks for gaze tracking
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """
        Takes a BGR frame (OpenCV), converts it to RGB (MediaPipe),
        and returns the processing results.
        """
        # 1. Convert BGR to RGB
        frame.flags.writeable = False # Optimization
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 2. Process the image
        results = self.face_mesh.process(rgb_frame)
        
        # 3. Cleanup
        frame.flags.writeable = True
        
        return results