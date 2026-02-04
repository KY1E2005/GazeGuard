import cv2
import sys
import config
import mediapipe as mp
import pygame 
import os
import numpy as np
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
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    phone_sound_path = os.path.join(assets_dir, "alert.wav")
    beep_sound_path = os.path.join(assets_dir, "beep.wav")
    
    try:
        phone_sound = pygame.mixer.Sound(phone_sound_path)
    except FileNotFoundError:
        phone_sound = None
        
    try:
        beep_sound = pygame.mixer.Sound(beep_sound_path)
    except FileNotFoundError:
        print(f"Warning: beep.wav not found at {beep_sound_path}")
        beep_sound = None
    was_phone_detected = False
    is_beep_playing = False
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    print("System Active. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        
        ui_state['width'] = width
        ui_state['height'] = height
        font_scale = width / 640.0

        # 1. ENVIRONMENT CHECK: BRIGHTNESS
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 60:
            if is_beep_playing and beep_sound:
                beep_sound.stop()
                is_beep_playing = False
            if was_phone_detected and phone_sound:
                phone_sound.stop()
                was_phone_detected = False
            text1 = "GazeGuard disabled due to poor lighting"
            text2 = "Please find suitable lighting conditions"
            t1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, 2)[0]
            t2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * font_scale, 2)[0]
            cv2.putText(frame, text1, ((width - t1_size[0]) // 2, height // 2 - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 165, 255), 2)
            cv2.putText(frame, text2, ((width - t2_size[0]) // 2, height // 2 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6 * font_scale, (0, 165, 255), 2)
            
            cv2.imshow(window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
            continue 
        
        # --- PHONE DETECTION ---
        phone_detected, phone_box = detector.detect(frame)
        
        if phone_detected:
            if not was_phone_detected and phone_sound:
                phone_sound.play() 
        else:
            if was_phone_detected and phone_sound:
                phone_sound.stop()
        was_phone_detected = phone_detected
        
        # --- FACE TRACKING ---
        results = tracker.process_frame(frame)
        
        should_play_beep = False
        
        # 3. NO FACE DETECTED CHECK
        if not results.multi_face_landmarks:
            # Stop distraction beep (Phone beep stays if phone is visible)
            if is_beep_playing and beep_sound:
                beep_sound.stop()
                is_beep_playing = False
            
            text = "No Face detected"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * font_scale, 2)[0]
            cv2.putText(frame, text, ((width - text_size[0]) // 2, height // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0 * font_scale, (0, 165, 255), 2)

        else:
            # 4. FACES FOUND - PROCESS MAIN USER
            all_faces = results.multi_face_landmarks
            num_faces = len(all_faces)
            # Sort by center proximity
            screen_center_x = 0.5 
            sorted_faces = []
            for face in all_faces:
                nose_x = face.landmark[1].x
                dist = abs(nose_x - screen_center_x)
                sorted_faces.append((face, dist, nose_x))
            sorted_faces.sort(key=lambda x: x[1])
            
            main_face = sorted_faces[0][0]
            main_nose_x = sorted_faces[0][2]
            
            # --- CENTERING CHECK ---
            if main_nose_x < 0.2 or main_nose_x > 0.8:
                should_play_beep = True
                warn_text = "Please center yourself for better accuracy"
                w_size = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, 2)[0]
                cv2.putText(frame, warn_text, ((width - w_size[0]) // 2, height - 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 165, 255), 2)

            # --- MULTIPLE FACE LOGIC ---
            if num_faces > 1:
                should_play_beep = True
                text = "MULTIPLE FACE DETECTED"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * font_scale, 2)[0]
                text_x = (width - text_size[0]) // 2
                text_y = int(height * 0.15)
                
                cv2.rectangle(frame, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
                cv2.putText(frame, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0 * font_scale, (0, 0, 255), 2)

            # --- SCORING ---
            landmarks = main_face.landmark
            ear = features.get_EAR(landmarks)
            gaze = features.get_gaze_ratio(landmarks)
            pitch, yaw, roll = features.get_head_pose(landmarks, frame.shape)
            
            score, duration, status = scorer.get_score(
                ear, gaze, pitch, yaw, phone_detected
            )
            
            # Long Distraction Logic
            if duration > 5.0:
                should_play_beep = True
                text = "Stay Focus"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * font_scale, 2)[0]
                text_x = (width - text_size[0]) // 2
                text_y = int(height * 0.3) 
                cv2.putText(frame, text, (text_x, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2 * font_scale, (0, 165, 255), 2)
            
            # --- VISUALS ---
            color = (0, 255, 0) if score > 50 else (0, 0, 255)
            cv2.putText(frame, f"Score: {score}%", (30, int(50 * font_scale)), 
                       cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2)
            status_text = f"Status: {status}"
            status_pos = (30, int(90 * font_scale))
            # Layer 1: Outline (Thick White)
            cv2.putText(frame, status_text, status_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (255, 255, 255), 5)
            # Layer 2: Main Text (Thin Black)
            cv2.putText(frame, status_text, status_pos, 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 0, 0), 2)
            if ui_state['debug_mode']:
                cv2.putText(frame, f"Gaze: {gaze:.2f} | Pitch: {pitch:.0f}", (30, height - 30),
                           cv2.FONT_HERSHEY_PLAIN, 1.0 * font_scale, (255, 255, 0), 1)

                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=main_face,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=main_face,
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

        # --- BEEP MANAGER ---
        if should_play_beep:
            if not is_beep_playing and beep_sound:
                beep_sound.play(loops=-1)
                is_beep_playing = True
        else:
            if is_beep_playing and beep_sound:
                beep_sound.stop()
                is_beep_playing = False

        # --- CHECKBOX UI ---
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