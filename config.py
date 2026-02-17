#For Settings Tweaking. Sets up the camera resolution or how strict the distraction rules are.

# --- Camera Settings ---
CAMERA_INDEX = 0   # 0 is the default webcam.
FRAME_WIDTH = 640  # Lower resolution = faster processing
FRAME_HEIGHT = 480

# --- Scoring Weights ---
WEIGHT_GAZE = 0.4
WEIGHT_POSE = 0.4
WEIGHT_BLINK = 0.2

# --- Visual Colors (BGR Format) ---
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)

# --- Calibration Data ---
EAR_THRESHOLD = 0.20          # Below this = BLINK
GAZE_LEFT_THRESHOLD = 0.35    # Below this = Looking LEFT
GAZE_RIGHT_THRESHOLD = 2.2   # Above this = Looking RIGHT
PITCH_THRESHOLD = 15  # Up/Down limit
YAW_THRESHOLD = 20    # Left/Right limit

# --- Scoring Weights ---
WEIGHT_GAZE = 0.4
WEIGHT_POSE = 0.4
WEIGHT_BLINK = 0.2