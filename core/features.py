import numpy as np
import cv2

class FeatureExtractor:
    def __init__(self):
        # --- Landmark Indices ---
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # 3D Model Points
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
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
        """Calculates iris position RELATIVE to the eye corners.
        Returns: ~0.5 (Center), <0.4 (Left), >0.6 (Right)"""
        def get_eye_ratio(eye_indices, iris_index):
            corner_A = np.array([landmarks[eye_indices[0]].x, landmarks[eye_indices[0]].y])
            corner_B = np.array([landmarks[eye_indices[3]].x, landmarks[eye_indices[3]].y])
            iris = np.array([landmarks[iris_index].x, landmarks[iris_index].y])
            eye_width = np.linalg.norm(corner_A - corner_B)
            if eye_width == 0: return 0.5
            dist_to_A = np.linalg.norm(iris - corner_A)
            return dist_to_A / eye_width

        ratio_left = get_eye_ratio(self.LEFT_EYE, 468)
        ratio_right = get_eye_ratio(self.RIGHT_EYE, 473)
        return (ratio_left + ratio_right) / 2.0

    def get_head_pose(self, landmarks, frame_shape):
        img_h, img_w, _ = frame_shape
        face_2d = []
        for idx in [1, 152, 263, 33, 291, 61]:
            face_2d.append([landmarks[idx].x * img_w, landmarks[idx].y * img_h])
        
        face_2d = np.array(face_2d, dtype=np.float64)
        focal_length = 1 * img_w
        cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
        dist_matrix = np.zeros((4, 1), dtype=np.float64)
        
        success, rot_vec, trans_vec = cv2.solvePnP(self.model_points, face_2d, cam_matrix, dist_matrix)
        rmat, jac = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        
        # Return degrees
        return angles[0], angles[1], angles[2]