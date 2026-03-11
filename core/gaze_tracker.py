#handle all the interaction with MediaPipe

import cv2
import mediapipe as mp

class GazeTracker:
    """A dedicated handler class for the MediaPipe FaceMesh model.
    Responsible for initializing the model and processing individual video frames 
    to extract 3D facial landmarks and iris coordinates."""
    
    def __init__(self):
        print("Loading MediaPipe FaceMesh...")
        self.mp_face_mesh = mp.solutions.face_mesh # Load the MediaPipe Face Mesh solution
        # Initialize the FaceMesh model with strict parameters
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame):
        """Takes a BGR frame (OpenCV), converts it to RGB (MediaPipe),
        and returns the processing results."""
        frame.flags.writeable = False # Pre-Processing & Optimization
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
        
        #  Process the image
        results = self.face_mesh.process(rgb_frame)
        
        #  Cleanup
        frame.flags.writeable = True
        
        return results