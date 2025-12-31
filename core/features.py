import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self):
        # --- Landmark Indices ---
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # --- 3D Model Points (Generic Human Face) ---
        # Nose tip, Chin, Left Eye corner, Right Eye corner, Mouth left, Mouth right
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ])

    def get_EAR(self, landmarks):
        def get_coords(indices):
            return np.array([(landmarks[i].x, landmarks[i].y) for i in indices])

        left_ear = self._calculate_eye_ratio(get_coords(self.LEFT_EYE))
        right_ear = self._calculate_eye_ratio(get_coords(self.RIGHT_EYE))
        return (left_ear + right_ear) / 2.0

    def _calculate_eye_ratio(self, eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        if C == 0: return 0.0
        return (A + B) / (2.0 * C)

    def get_gaze_ratio(self, landmarks):
        """
        Calculates the iris position relative to the eye corners.
        Returns: A value roughly between 0.0 (Looking Left) and 1.0 (Looking Right).
        Center is typically ~0.5.
        """
        def get_eye_ratio(eye_indices, iris_index):
            # Get key points: Inner Corner, Outer Corner, and Iris
            # eye_indices[0] and eye_indices[3] are the corners in standard MediaPipe 
            corner_A = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
            corner_B = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
            iris = np.array([landmarks[iris_index].x, landmarks[iris_index].y])
            
            # Calculate the total eye width
            eye_width = np.linalg.norm(corner_A - corner_B)
            if eye_width == 0: return 0.5
            
            # Calculate distance from one corner to the iris
            dist_to_A = np.linalg.norm(iris - corner_A)
            
            # The ratio (0.0 to 1.0)
            return dist_to_A / eye_width

        # Average both eyes for stability
        # Left Eye (indices 33, 133) & Left Iris (468)
        # Right Eye (indices 362, 263) & Right Iris (473)
        ratio_left = get_eye_ratio(self.LEFT_EYE, 468)
        ratio_right = get_eye_ratio(self.RIGHT_EYE, 473)
        
        return (ratio_left + ratio_right) / 2.0

    def get_head_pose(self, landmarks, frame_shape):
        """
        Estimates Head Pose (Pitch, Yaw, Roll) using SolvePnP.
        Returns: (pitch, yaw, roll) in degrees.
        """
        img_h, img_w, _ = frame_shape
        
        # 1. Get 2D Image Points
        face_2d = []
        face_2d.append([landmarks[1].x * img_w, landmarks[1].y * img_h])      # Nose tip
        face_2d.append([landmarks[152].x * img_w, landmarks[152].y * img_h])  # Chin
        face_2d.append([landmarks[263].x * img_w, landmarks[263].y * img_h])  # Left eye corner
        face_2d.append([landmarks[33].x * img_w, landmarks[33].y * img_h])    # Right eye corner
        face_2d.append([landmarks[291].x * img_w, landmarks[291].y * img_h])  # Mouth Left
        face_2d.append([landmarks[61].x * img_w, landmarks[61].y * img_h])    # Mouth Right
        
        face_2d = np.array(face_2d, dtype=np.float64)

        # 2. Camera Matrix (Approximation)
        focal_length = 1 * img_w
        cam_matrix = np.array([
            [focal_length, 0, img_h / 2],
            [0, focal_length, img_w / 2],
            [0, 0, 1]
        ])
        
        # 3. Solve PnP
        # Assuming no lens distortion (zeros)
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        success, rot_vec, trans_vec = cv2.solvePnP(
            self.model_points, face_2d, cam_matrix, dist_matrix
        )

        # 4. Convert Rotation Vector to Angles (Euler)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, mtxR, mtxQ, Q, Qx, Qy = cv2.RQDecomp3x3(rmat)

        # angles[0] = Pitch (Up/Down)
        # angles[1] = Yaw (Left/Right)
        # angles[2] = Roll (Tilt)
        return angles[0], angles[1], angles[2]