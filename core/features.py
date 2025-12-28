#Turn raw face landmarks to numbers
import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self):
        # --- Landmark Indices (MediaPipe Face Mesh Standard) ---
        # specific points on the face mesh that correspond to the eyes
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]

    def get_EAR(self, landmarks):
        """
        Calculates Eye Aspect Ratio (EAR) to detect blinking.
        Formula: (dist_vertical_1 + dist_vertical_2) / (2 * dist_horizontal)
        """
        # Helper to get coordinates
        def get_coords(indices):
            return np.array([(landmarks[i].x, landmarks[i].y) for i in indices])

        # Calculate for both eyes
        left_ear = self._calculate_eye_ratio(get_coords(self.LEFT_EYE))
        right_ear = self._calculate_eye_ratio(get_coords(self.RIGHT_EYE))

        # Return average EAR (handling both eyes gives better stability)
        return (left_ear + right_ear) / 2.0

    def _calculate_eye_ratio(self, eye_points):
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        if C == 0: return 0.0 # Prevent division by zero
        return (A + B) / (2.0 * C)

    def get_gaze_ratio(self, landmarks):
        """
        Calculates the horizontal position of the iris.
        Returns: A value between 0.0 (Looking Left) and 1.0 (Looking Right).
        0.5 is perfectly Center.
        """
        left_iris = landmarks[468]  # Center of left iris
        right_iris = landmarks[473] # Center of right iris
        
        # "Gaze" is the average horizontal position of both irises
        avg_gaze_x = (left_iris.x + right_iris.x) / 2.0
        return avg_gaze_x