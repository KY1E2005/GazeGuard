from flask import Flask, render_template, Response, request, send_from_directory, jsonify
import cv2
import config
import mediapipe as mp
import numpy as np
import os
import tempfile
import time
import pygame

from core.gaze_tracker import GazeTracker
from core.features import FeatureExtractor
from core.scorer import ConcentrationScorer
from core.detector import ObjectDetector

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'gazeguard_uploads')
RECORDS_FOLDER = os.path.join(tempfile.gettempdir(), 'gazeguard_records')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECORDS_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RECORDS_FOLDER'] = RECORDS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024 # 200MB limit for video uploads

# visual state
ui_state = {
    'show_landmarks': True
}

# --- GLOBAL STATUS STATE ---
current_system_status = {
    'brightness': 0,
    'lighting_ok': False,
    'face_detected': False,
    'centered': False,
    'score': 0,           
    'status': "Unknown",    
    'beep_active': False,   
    'phone_detected': False,
    'multiple_faces': False
}

# --- AUDIO SETUP ---
pygame.mixer.init()
assets_dir = os.path.join(os.path.dirname(__file__), "assets")
try: phone_sound = pygame.mixer.Sound(os.path.join(assets_dir, "alert.wav"))
except: phone_sound = None
try: beep_sound = pygame.mixer.Sound(os.path.join(assets_dir, "beep.wav"))
except: beep_sound = None

# --- CORE CLASSES ---
tracker = GazeTracker()
features = FeatureExtractor()
scorer = ConcentrationScorer()
detector = ObjectDetector()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

camera = None

def get_camera():
    """Initializes and returns the webcam instance based on config parameters."""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(config.CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return camera

def generate_frames():
    """The core video processing generator. Captures frames from the webcam, runs them 
    through the AI models, updates the global state, and yields JPEG frames to the browser."""
    def create_error_frame(message):
        """Helper to generate a visual error screen if the camera fails."""
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 + text_size[1]) // 2
        cv2.putText(blank_frame, message, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', blank_frame)
        return (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cam = get_camera()
    
    was_phone_detected = False
    is_beep_playing = False
    look_away_start = None
    
    global current_system_status
    try:
        while True:
            # SAFETY CHECK: If camera failed to open
            if cam is None or not cam.isOpened():
                current_system_status.update({
                    'lighting_ok': False, 'face_detected': False, 'centered': False, 'brightness': 0,
                    'score': 0, 'status': "Camera Disconnected"
                })
                yield create_error_frame("Camera Disconnected!")
                time.sleep(1)
                cam = get_camera()
                continue
                
            success, frame = cam.read()
            
            # SAFETY CHECK: If camera is open but returning no frames (Privacy Shutter)
            if not success:
                current_system_status.update({
                    'lighting_ok': False, 'face_detected': False, 'centered': False, 'brightness': 0,
                    'score': 0, 'status': "Camera Blocked"
                })
                yield create_error_frame("Camera Blocked/Unavailable")
                cam.release()
                time.sleep(1)
                cam = get_camera()
                continue
            
            frame = cv2.flip(frame, 1) # Mirror image for natural user experience
            height, width, _ = frame.shape
            font_scale = width / 640.0
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            global_brightness = np.mean(gray)
            # Run background AI tasks
            phone_detected, phone_box = detector.detect(frame)
            results = tracker.process_frame(frame)

            # Determine basic calibration status
            is_centered = False
            if results.multi_face_landmarks:
                face_x = results.multi_face_landmarks[0].landmark[1].x 
                is_centered = 0.3 < face_x < 0.7 

            current_system_status.update({
                'brightness': int(global_brightness),
                'lighting_ok': bool(global_brightness > 50),
                'face_detected': bool(results.multi_face_landmarks),
                'centered': bool(is_centered)
            })

            should_play_beep = False
            system_active = False
            current_action = "None"
            multi_face_flag = False
            score = 0 
            current_status_text = "Unknown" 
            
            # --- LOGIC BRANCH 1: No Face Detected ---
            if not results.multi_face_landmarks:
                if global_brightness < 50:
                    current_status_text = "Too Dark" # Assign dark state 
                    text1 = "GazeGuard disabled due to poor lighting"
                    text2 = "Please find suitable lighting conditions"
                    t1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, 2)[0]
                    t2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * font_scale, 2)[0]
                    cv2.putText(frame, text1, ((width - t1_size[0]) // 2, height // 2 - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 165, 255), 2)
                    cv2.putText(frame, text2, ((width - t2_size[0]) // 2, height // 2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6 * font_scale, (0, 165, 255), 2)
                else:
                    current_status_text = "No Face" # Assign no face state
                    text = "No Face detected"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * font_scale, 2)[0]
                    cv2.putText(frame, text, ((width - text_size[0]) // 2, height // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0 * font_scale, (0, 165, 255), 2)
                
                look_away_start = None
                if is_beep_playing and beep_sound:
                    beep_sound.stop()
                    is_beep_playing = False

            # --- LOGIC BRANCH 2: Faces Detected ---
            else:
                all_faces = results.multi_face_landmarks
                screen_center_x = 0.5 
                sorted_faces = []
                for face in all_faces:
                    nose_x = face.landmark[1].x
                    dist = abs(nose_x - screen_center_x)
                    sorted_faces.append((face, dist, nose_x))
                sorted_faces.sort(key=lambda x: x[1])
                main_face = sorted_faces[0][0]
                main_nose_x = sorted_faces[0][2]

                # Dynamic Face Brightness (Ensures the face itself is illuminated, not just the background)
                try:
                    h_img, w_img = gray.shape
                    x_min = int(min([l.x for l in main_face.landmark]) * w_img)
                    x_max = int(max([l.x for l in main_face.landmark]) * w_img)
                    y_min = int(min([l.y for l in main_face.landmark]) * h_img)
                    y_max = int(max([l.y for l in main_face.landmark]) * h_img)
                    x_min, x_max = max(0, x_min), min(w_img, x_max)
                    y_min, y_max = max(0, y_min), min(h_img, y_max)
                    face_roi = gray[y_min:y_max, x_min:x_max]
                    face_brightness = np.mean(face_roi) if face_roi.size > 0 else 0
                except:
                    face_brightness = global_brightness

                if face_brightness < 40:
                    current_status_text = "Face Too Dark" 
                    text = "Face too dark for accuracy"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8 * font_scale, 2)[0]
                    cv2.putText(frame, text, ((width - text_size[0]) // 2, height // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8 * font_scale, (0, 165, 255), 2)
                    look_away_start = None
                else:
                    system_active = True
                    num_faces = len(all_faces)

                    if main_nose_x < 0.2 or main_nose_x > 0.8:
                        should_play_beep = True
                        warn_text = "Please center yourself for better accuracy"
                        w_size = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * font_scale, 2)[0]
                        cv2.putText(frame, warn_text, ((width - w_size[0]) // 2, height - 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6 * font_scale, (0, 165, 255), 2)

                    if num_faces > 1:
                        multi_face_flag = True
                        should_play_beep = True
                        text = "MULTIPLE FACES DETECTED"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * font_scale, 2)[0]
                        text_x = (width - text_size[0]) // 2
                        text_y = int(height * 0.15)
                        cv2.rectangle(frame, (text_x - 10, text_y - 30), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 0), -1)
                        cv2.putText(frame, text, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.0 * font_scale, (0, 0, 255), 2)

                    landmarks = main_face.landmark
                    ear = features.get_EAR(landmarks)
                    gaze = features.get_gaze_ratio(landmarks)
                    pitch, yaw, roll = features.get_head_pose(landmarks, frame.shape)
                    mar = features.get_MAR(landmarks)
                    
                    score, duration, status = scorer.get_score(ear, gaze, pitch, yaw, phone_detected)
                    current_status_text = status # Inherit string from scorer
                    
                    # --- BALANCED LOOK AWAY LOGIC ---
                    is_looking_down_severe = pitch > 15
                    is_looking_down_mild = pitch > 8
                    is_looking_up = pitch < -25
                    is_turning = abs(yaw) > 20      
                    severe_pose = is_looking_down_severe or is_looking_up or (abs(yaw) > 35)
                    severe_gaze = (gaze < 0.35) or (gaze > 0.65)
                    moderate_pose = is_looking_down_mild or is_turning or (abs(roll) > 20)
                    drifting_gaze = (gaze < 0.42) or (gaze > 0.58)
                    is_looking_away = severe_pose or severe_gaze or (moderate_pose and drifting_gaze)

                    if moderate_pose and not is_looking_away:
                        warn_text = "Please keep your head straight"
                        w_size = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, 2)[0]
                        cv2.putText(frame, warn_text, ((width - w_size[0]) // 2, height - 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 255, 255), 2)
                        
                    if is_looking_away:
                        if look_away_start is None:
                            look_away_start = time.time()
                        threshold_time = 3.0 if is_looking_down_severe else 4.0
                        if time.time() - look_away_start > threshold_time:
                            should_play_beep = True
                            cv2.putText(frame, "Stay Focus", (int(width/2 - 100), int(height * 0.3)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2 * font_scale, (0, 165, 255), 2)
                    else:
                        look_away_start = None

                    if duration > 3.0:
                        should_play_beep = True
                        cv2.putText(frame, "Stay Focus", (int(width/2 - 100), int(height * 0.3)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2 * font_scale, (0, 165, 255), 2)
                        
                        # --- MAR ACTION LOGIC (Yawning/Talking) ---
                    if mar > 0.45:
                        current_action = "Yawning"
                    elif mar > 0.25:
                        current_action = "Talking/Smiling"
                        
                    if current_action != "None":
                        cv2.putText(frame, f"Action: {current_action}", (30, int(130 * font_scale)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 165, 255), 2)

                    # --- VISUALS TOGGLE ---
                    if ui_state['show_landmarks']:
                        # Draw Mesh
                        mp_drawing.draw_landmarks(
                            image=frame, landmark_list=main_face,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        # Draw Irises
                        mp_drawing.draw_landmarks(
                            image=frame, landmark_list=main_face,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                        # Draw Debug Data
                        cv2.putText(frame, f"Gaze: {gaze:.2f} | Pitch: {pitch:.0f} | MAR: {mar:.2f}", 
                                    (30, height - 30), cv2.FONT_HERSHEY_PLAIN, 1.0 * font_scale, (255, 255, 0), 1)

                    color = (0, 255, 0) if score > 50 else (0, 0, 255)
                    cv2.putText(frame, f"Score: {score}%", (30, int(50 * font_scale)), 
                               cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2)
                    
                    status_text = f"Status: {status}"
                    cv2.putText(frame, status_text, (30, int(90 * font_scale)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 0, 0), 5)
                    cv2.putText(frame, status_text, (30, int(90 * font_scale)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (255, 255, 255), 2)

            # --- AUDIO ALERTS & CONTROLS ---
            if phone_detected:
                 if phone_box: 
                     bx, by, bw, bh = phone_box
                     cv2.rectangle(frame, (int(bx), int(by)), (int(bx + bw), int(by + bh)), (0, 0, 255), 3)

                 cv2.putText(frame, "PHONE DETECTED", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                 if not was_phone_detected and phone_sound:
                     phone_sound.play()
            else:
                 if was_phone_detected and phone_sound:
                     phone_sound.stop()
            was_phone_detected = phone_detected

            if system_active:
                if should_play_beep:
                    if not is_beep_playing and beep_sound:
                        beep_sound.play(loops=-1)
                        is_beep_playing = True
                else:
                    if is_beep_playing and beep_sound:
                        beep_sound.stop()
                        is_beep_playing = False
            else:
                if is_beep_playing and beep_sound:
                    beep_sound.stop()
                    is_beep_playing = False

            # --- UPDATE GLOBAL STATUS FOR FRONTEND ---
            current_system_status['score'] = int(score) if system_active else 0
            current_system_status['status'] = current_status_text
            current_system_status['beep_active'] = is_beep_playing
            current_system_status['phone_detected'] = phone_detected
            current_system_status['mar_action'] = current_action
            current_system_status['multiple_faces'] = multi_face_flag
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        if is_beep_playing and beep_sound: beep_sound.stop()
        if phone_sound: phone_sound.stop()

# --- FLASK ROUTES & API ENDPOINTS ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/session')
def session_page():
    return render_template('session.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown_camera')
def shutdown_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None
    pygame.mixer.stop()
    return "Camera Stopped"

@app.route('/toggle_visual/<feature>')
def toggle_visual(feature):
    if feature in ui_state:
        ui_state[feature] = not ui_state[feature]
        return jsonify({'status': 'ok', 'state': ui_state[feature]})
    return jsonify({'status': 'error'}), 400

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    # Retrieve a list of files.
    files = request.files.getlist('file')
    if not files:
        return jsonify({'error': 'No file part'}), 400
    
    urls = []
    for file in files:
        if file.filename != '':
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            urls.append(f'/uploads/{filename}')
            
    if not urls:
        return jsonify({'error': 'No selected files'}), 400
        
    return jsonify({'urls': urls})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/assets/<filename>')
def serve_assets(filename):
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    return send_from_directory(assets_dir, filename)

@app.route('/get_status')
def get_status():
    return jsonify(current_system_status)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)