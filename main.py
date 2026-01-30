import cv2
import sys
import config
import mediapipe as mp
import pygame 
import os     
from core.gaze_tracker import GazeTracker
from core.features import FeatureExtractor
from core.scorer import ConcentrationScorer
from core.detector import ObjectDetector

# --- Global State for UI Toggle ---
ui_state = {
    'debug_mode': True,
    'width': 640,       
    'height': 480
}

# --- Mouse Callback ---
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        w, h = ui_state['width'], ui_state['height']
        x_min = w - 160
        y_min = h - 70
        x_max = w
        y_max = h
        
        if x_min <= x <= x_max and y_min <= y <= y_max:
            ui_state['debug_mode'] = not ui_state['debug_mode']
            print(f"Visuals Toggled: {ui_state['debug_mode']}")

def main():
    # 1. Setup Camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit()

    # 2. Setup Window & Mouse Callback
    window_name = "GazeGuard"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    # 3. Initialize Classes
    tracker = GazeTracker()
    features = FeatureExtractor()
    scorer = ConcentrationScorer()
    detector = ObjectDetector()
    
    # --- Audio Setup ---
    pygame.mixer.init()
    sound_path = os.path.join(os.path.dirname(__file__), "assets", "alert.wav")
    try:
        alert_sound = pygame.mixer.Sound(sound_path)
    except FileNotFoundError:
        print(f"Warning: Sound file not found at {sound_path}. Audio disabled.")
        alert_sound = None 

    was_phone_detected = False
    
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
        
        height, width, _ = frame.shape
        ui_state['width'] = width
        ui_state['height'] = height
        font_scale = width / 640.0

        # --- PHONE DETECTION ---
        phone_detected, phone_box = detector.detect(frame)
        
        # --- Audio Logic ---
        if phone_detected:
            if not was_phone_detected and alert_sound:
                alert_sound.play() 
        else:
            if was_phone_detected and alert_sound:
                alert_sound.stop()
        
        was_phone_detected = phone_detected
        
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
                color = (0, 255, 0) if score > 50 else (0, 0, 255)
                
                # Score
                cv2.putText(frame, f"Score: {score}%", (30, int(50 * font_scale)), 
                           cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2)
                # Status
                cv2.putText(frame, f"Status: {status}", (30, int(90 * font_scale)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (200, 200, 200), 2)
                
                # --- CONDITIONAL VISUALS ---
                if ui_state['debug_mode']:
                    # Debug Text (Bottom Left)
                    cv2.putText(frame, f"Gaze: {gaze:.2f} | Pitch: {pitch:.0f}", (30, height - 30),
                               cv2.FONT_HERSHEY_PLAIN, 1.0 * font_scale, (255, 255, 0), 1)

                    # Face Mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    # Iris Mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                    )

                # --- Phone Warnings ---
                if phone_detected:
                    if phone_box and ui_state['debug_mode']: 
                        bx, by, bw, bh = phone_box
                        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 3)

                    text = "PHONE DETECTED"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5 * font_scale, 3)[0]
                    text_x = (width - text_size[0]) // 2
                    text_y = (height + text_size[1]) // 2
                    cv2.putText(frame, text, (text_x, text_y), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1.5 * font_scale, (0, 0, 255), 3)

        # --- DRAW CHECKBOX UI ---
        margin_right = 40
        margin_bottom = 30
        box_size = 20
        box_x = width - margin_right - box_size
        box_y = height - margin_bottom - box_size
        
        text_label = "Hide Visuals"
        cv2.putText(frame, text_label, (box_x - 40, box_y - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (255, 255, 255), 1)
        
        if not ui_state['debug_mode']:
            cv2.line(frame, (box_x, box_y), (box_x + box_size, box_y + box_size), (0, 255, 0), 2)
            cv2.line(frame, (box_x, box_y + box_size), (box_x + box_size, box_y), (0, 255, 0), 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()