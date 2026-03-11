# GazeGuard: Computer Vision Concentration Tracker

## Overview

GazeGuard is a web-based and standalone desktop concentration tracking system designed to monitor student focus and ensure academic integrity during remote learning. It utilizes advanced computer vision to track eye aspect ratio (EAR), gaze ratio, head pose (pitch/yaw/roll), and mouth aspect ratio (MAR). It also features an AI object detector to flag mobile phone usage and generates a cryptographically verifiable "Proof of Study" PDF report.

Privacy-First Architecture: All video processing and AI inference are performed entirely locally on the user's device. No video footage is ever transmitted to a cloud server.

## Features
- Real-time Facial Tracking: Monitors blinking, drowsiness, and head orientation.
- Anti-Cheat & Environment Checks: Dynamically verifies adequate lighting, enforces camera centering, and triggers alerts if multiple faces appear in the frame.
- AI Object Detection: Instantly flags mobile phone usage using TensorFlow Lite.
- Exam Mode: Provides a secure answer environment and generates a locked cryptographic word sequence to prevent post-session answer tampering.
- Comprehensive Data Analytics: Offers an interactive, zoomable dashboard to review historical sessions, analyze concentration trends, and export raw data to .json or .csv.
- Verifiable PDF Reports: Generates official session reports with data charts and QR code verification.
- Local Storage: Stores session history and video recordings directly in the browser using IndexedDB.

## Prerequisites
- To run this project, you will need **Python 3.8 or higher** installed on your system.
- Modern web browser (Chrome or Edge recommended) for the web UI.
- You will also need a functional webcam connected to your device.

## Installation & Setup
1. Clone the repository or extract the project folder:  
```git clone https://github.com/KY1E2005/GazeGuard.git
cd GazeGuard```
2. Create a virtual environment (Recommended):
`python -m venv venv`
**On Windows:**
`venv\Scripts\activate`  
**On Mac/Linux:**
`source venv/bin/activate`
3. Install the required Python dependencies:  
`pip install Flask opencv-python mediapipe numpy pygame`
4. Ensure the AI Model is present:
Verify that the `efficientdet_lite0.tflite` model file is located in the root directory (or inside the core/ folder alongside detector.py).

## Run the Application

### Mode 1: Web Application (Flask Backend):
1. Open your terminal in the project root directory.
2. Run the Flask server:
`python app.py`

### Mode 2: Standalone Desktop App (OpenCV Native):
This mode bypasses the web browser and runs a lightweight desktop window strictly for testing the tracking and drawing algorithms.
1. Open your terminal in the project root directory.
2. Run the main script:
`python main.py`
3. Click inside the video window to toggle the debug visuals (mesh and bounding boxes) on and off. Press **Q** to quit.

## Acknowledgements & Code Provenance
**1. Backend Libraries & AI Models:**
- MediaPipe (Google): Used for real-time 3D face meshing, iris tracking, and facial landmark extraction (gaze_tracker.py, features.py).
- TensorFlow Lite (Google): The efficientdet_lite0.tflite model is an open-source model used inside detector.py for mobile phone detection.
- OpenCV (cv2): Used for image matrix manipulation, SolvePnP (head pose estimation), and drawing debug visuals.
- Flask & Pygame: Used for backend routing and local audio playback, respectively.

**2. Frontend Libraries (CDNs)**  
The following JavaScript libraries were included via CDN in the HTML templates for UI and data visualization:
- Chart.js & chartjs-plugin-zoom: Used to render the interactive concentration graphs in dashboard.html.
- jsPDF & html2canvas: Used to capture the dashboard analytics and generate the downloadable "Proof of Study" PDF report.
- QRcode-generator: Used to generate the cryptographic verification QR codes appended to the final reports.
- Hammer.js: Used to support touch gestures for the zoomable chart interface.

**3. External Code Snippets**  
Custom UI Components (CSS/HTML): Several interactive UI elements—including various buttons, modals, input fields, and the custom +/- number input for the Pomodoro timer used in session.html and dashboard.html—were adapted from open-source design snippets originally sourced on (https://uiverse.io/). The base CSS for these components was extensively modified to match GazeGuard's custom dark theme and integrated with specific JavaScript logic.

**4. Media Assets**  
Audio Alerts: The chime.wav (Pomodoro completion sound) was sourced from a free sound effects library (https://freesound.org/). The beep.wav, alert.wav, and chime.wav files are standard open-source audio assets used for distraction warnings. 