from flask import Flask, render_template, Response, request, send_from_directory, jsonify
import cv2
import config
import mediapipe as mp
import numpy as np
import os
import tempfile 

from core.gaze_tracker import GazeTracker
from core.features import FeatureExtractor
from core.scorer import ConcentrationScorer
from core.detector import ObjectDetector

app = Flask(__name__)

# --- CONFIGURATION ---
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), 'gazeguard_uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ui_state = {
    'debug_mode': True
}

# --- CORE CLASSES ---
tracker = GazeTracker()
features = FeatureExtractor()
scorer = ConcentrationScorer()
detector = ObjectDetector()

# --- MEDIAPIPE DRAWING ---
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
    while True:
        success, frame = cam.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        font_scale = width / 640.0
        phone_detected, phone_box = detector.detect(frame)
        results = tracker.process_frame(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # --- METRICS ---
                landmarks = face_landmarks.landmark
                ear = features.get_EAR(landmarks)
                gaze = features.get_gaze_ratio(landmarks)
                pitch, yaw, roll = features.get_head_pose(landmarks, frame.shape)
                score, duration, status = scorer.get_score(ear, gaze, pitch, yaw, phone_detected)
                
                if ui_state['debug_mode']:
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
             cv2.putText(frame, "PHONE DETECTED", (50, 240), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shutdown_camera')
def shutdown_camera():
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None
    return "Camera Stopped"

@app.route('/toggle_visuals')
def toggle_visuals():
    ui_state['debug_mode'] = not ui_state['debug_mode']
    return jsonify({'status': 'ok', 'mode': ui_state['debug_mode']})

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return jsonify({'url': f'/uploads/{filename}'})

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)