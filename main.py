#This is the entry point of GazeGuard.
import cv2
import sys
import config
import mediapipe as mp
from core.gaze_tracker import GazeTracker
from core.features import FeatureExtractor
from core.scorer import ConcentrationScorer

def main():
    # 1. Setup Camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    # 2. Initialize the Tracker
    tracker = GazeTracker()
    features = FeatureExtractor()
    scorer = ConcentrationScorer()
    
    # Temporary drawing utility just for testing
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    print("System Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for mirror effect
        frame = cv2.flip(frame, 1)

        # --- CORE PROCESS ---
        results = tracker.process_frame(frame)

        # --- VISUALIZATION (Test) ---
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                
                # 1. Calculate Metrics
                ear = features.get_EAR(landmarks)
                gaze = features.get_gaze_ratio(landmarks)
                pitch, yaw, roll = features.get_head_pose(landmarks, frame.shape)
                
                # 2. Print to Terminal (Debug)
                # "f-string" formatting limits decimal places for readability
                print(f"EAR: {ear:.3f} | Gaze X: {gaze:.3f}")
                print(f"Pitch: {pitch:.1f} | Yaw: {yaw:.1f} | Roll: {roll:.1f}")
                score, duration, status = scorer.get_score(ear, gaze, pitch, yaw)
                
                color = (0, 255, 0) if score > 50 else (0, 0, 255)
                
                cv2.putText(frame, f"Score: {score}%", (30, 50), 
                           cv2.FONT_HERSHEY_DUPLEX, 1, color, 2)
                
                # Draw Status
                cv2.putText(frame, f"Status: {status}", (30, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                           
                # Debug Data
                cv2.putText(frame, f"Gaze: {gaze:.2f} | Yaw: {yaw:.0f}", (30, 450),
                           cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1)
                
                # Draw the mesh on the face
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                
                # Draw the eyes/iris (Validation that refine_landmarks is working)
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