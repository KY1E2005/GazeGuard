from flask import Flask, render_template, Response
import cv2
import config
from core.gaze_tracker import GazeTracker
from core.features import FeatureExtractor
from core.scorer import ConcentrationScorer
from core.detector import ObjectDetector
import os

app = Flask(__name__)
tracker = GazeTracker()
features = FeatureExtractor()
scorer = ConcentrationScorer()
detector = ObjectDetector()
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
        
        phone_detected, phone_box = detector.detect(frame)
        results = tracker.process_frame(frame)
        
        if phone_detected:
             cv2.putText(frame, "PHONE DETECTED", (50, 240), 
                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ear = features.get_EAR(face_landmarks.landmark)
                score, _, status = scorer.get_score(ear, 0.5, 0, 0, phone_detected)
                
                cv2.putText(frame, f"Score: {score}%", (30, 50), 
                           cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)

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
    """Optional: Endpoint to release camera when user stops the session"""
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None
    return "Camera Stopped"

if __name__ == "__main__":
    app.run(debug=True)