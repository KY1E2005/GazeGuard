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
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# visual state
ui_state = {
    'show_landmarks': True
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
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(config.CAMERA_INDEX)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return camera

def generate_frames():
    cam = get_camera()
    
    was_phone_detected = False
    is_beep_playing = False
    look_away_start = None

    try:
        while True:
            success, frame = cam.read()
            if not success:
                break
            
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            font_scale = width / 640.0

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            global_brightness = np.mean(gray)

            phone_detected, phone_box = detector.detect(frame)
            results = tracker.process_frame(frame)

            should_play_beep = False
            system_active = False

            if not results.multi_face_landmarks:
                if global_brightness < 60: 
                    text1 = "GazeGuard disabled due to poor lighting"
                    text2 = "Please find suitable lighting conditions"
                    t1_size = cv2.getTextSize(text1, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, 2)[0]
                    t2_size = cv2.getTextSize(text2, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * font_scale, 2)[0]
                    cv2.putText(frame, text1, ((width - t1_size[0]) // 2, height // 2 - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 165, 255), 2)
                    cv2.putText(frame, text2, ((width - t2_size[0]) // 2, height // 2 + 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6 * font_scale, (0, 165, 255), 2)
                else:
                    text = "No Face detected"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * font_scale, 2)[0]
                    cv2.putText(frame, text, ((width - text_size[0]) // 2, height // 2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0 * font_scale, (0, 165, 255), 2)
                
                look_away_start = None
                if is_beep_playing and beep_sound:
                    beep_sound.stop()
                    is_beep_playing = False

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
                    
                    score, duration, status = scorer.get_score(ear, gaze, pitch, yaw, phone_detected)
                    
                    # --- GAZE-DOMINANT LOOK AWAY LOGIC ---
                    bad_gaze = (gaze < 0.35) or (gaze > 0.65) # Eyes clearly off screen
                    bad_pose = (abs(pitch) > 25) or (abs(yaw) > 30) or (abs(roll) > 25) 
                    drifting_gaze = (gaze < 0.40) or (gaze > 0.60)
                    
                    # Only flag if eyes are gone, OR if head is heavily turned AND eyes are drifting
                    is_looking_away = bad_gaze or (bad_pose and drifting_gaze)

                    if bad_pose and not is_looking_away:
                        warn_text = "Please keep your head straight"
                        w_size = cv2.getTextSize(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, 2)[0]
                        cv2.putText(frame, warn_text, ((width - w_size[0]) // 2, height - 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 255, 255), 2)
                        
                    if is_looking_away:
                        if look_away_start is None:
                            look_away_start = time.time()
                        if time.time() - look_away_start > 5.0:
                            should_play_beep = True
                            cv2.putText(frame, "Stay Focus", (int(width/2 - 100), int(height * 0.3)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1.2 * font_scale, (0, 165, 255), 2)
                    else:
                        look_away_start = None

                    if duration > 5.0:
                        should_play_beep = True
                        cv2.putText(frame, "Stay Focus", (int(width/2 - 100), int(height * 0.3)), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2 * font_scale, (0, 165, 255), 2)

                    # --- SINGLE LANDMARKS TOGGLE ---
                    if ui_state['show_landmarks']:
                        mp_drawing.draw_landmarks(
                            image=frame, landmark_list=main_face,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=frame, landmark_list=main_face,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                        
                    cv2.putText(frame, f"Gaze: {gaze:.2f} | Pitch: {pitch:.0f}", (30, height - 30),
                                cv2.FONT_HERSHEY_PLAIN, 1.0 * font_scale, (255, 255, 0), 1)

                    color = (0, 255, 0) if score > 50 else (0, 0, 255)
                    cv2.putText(frame, f"Score: {score}%", (30, int(50 * font_scale)), 
                               cv2.FONT_HERSHEY_DUPLEX, font_scale, color, 2)
                    
                    status_text = f"Status: {status}"
                    cv2.putText(frame, status_text, (30, int(90 * font_scale)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 0, 0), 5)
                    cv2.putText(frame, status_text, (30, int(90 * font_scale)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (255, 255, 255), 2)

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

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
    finally:
        if is_beep_playing and beep_sound: beep_sound.stop()
        if phone_sound: phone_sound.stop()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/session')
def session_page():
    return render_template('session.html')

@app.route('/dashboard')
def dashboard():
    return """
    <html>
        <body style="font-family: Arial, sans-serif; text-align: center; padding-top: 100px; background-color: #1a1a1a; color: white;">
            <h1 style="color: #FD871F; font-size: 3rem;">Dashboard Overview</h1>
            <p style="font-size: 1.2rem; color: #a6a7aa; margin-bottom: 40px;">Post-Session Review and Analytics will be built here.</p>
            <a href="/" style="color: white; border: 2px solid #FD871F; padding: 15px 30px; text-decoration: none; border-radius: 5px; font-weight: bold;">Return Home</a>
        </body>
    </html>
    """

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

if __name__ == "__main__":
    app.run(debug=True)