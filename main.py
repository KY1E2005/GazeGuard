import cv2
import sys
import config
import mediapipe as mp
from core.gaze_tracker import GazeTracker
from core.features import FeatureExtractor
from core.scorer import ConcentrationScorer
from core.detector import ObjectDetector

def main():
    # 1. Setup Camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    # 2. Initialize Classes
    tracker = GazeTracker()
    features = FeatureExtractor()
    scorer = ConcentrationScorer()
    detector = ObjectDetector()
    
    # Visualization utils
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    print("System Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Flip frame
        frame = cv2.flip(frame, 1)
        
        # Get frame dimensions for dynamic sizing
        height, width, _ = frame.shape

        # --- PHONE DETECTION ---
        phone_detected = detector.detect(frame)
        
        # --- FACE TRACKING ---
        results = tracker.process_frame(frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # 1. Get Biometrics
                ear = features.get_EAR(landmarks)
                gaze = features.get_gaze_ratio(landmarks)
                pitch, yaw, roll = features.get_head_pose(landmarks, frame.shape)
                
                # 2. Get Score
                score, duration, status = scorer.get_score(
                    ear, gaze, pitch, yaw, phone_detected
                )
                
                # 3. Visuals
                font_scale = width / 640.0
                # Score Color
                color = (0, 255, 0) if score > 50 else (0, 0, 255)
                # Draw Score
                cv2.putText(frame, f"Score: {score}%", (30, int(50 * font_scale)), 
                           cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2)
                # Draw Status
                cv2.putText(frame, f"Status: {status}", (30, int(90 * font_scale)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (200, 200, 200), 2)
                # Debug Data (Bottom Left)
                cv2.putText(frame, f"Gaze: {gaze:.2f} | Pitch: {pitch:.0f}", (30, height - 30),
                           cv2.FONT_HERSHEY_PLAIN, 1.0 * font_scale, (255, 255, 0), 1)
                # Draw Phone Warning
                if phone_detected:
                    text = "PHONE DETECTED"
                    # Calculate center position
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5 * font_scale, 3)[0]
                    text_x = (width - text_size[0]) // 2
                    text_y = (height + text_size[1]) // 2
                    
                    cv2.putText(frame, text, (text_x, text_y), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1.5 * font_scale, (0, 0, 255), 3)

                # Draw Mesh
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        cv2.imshow("GazeGuard", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()