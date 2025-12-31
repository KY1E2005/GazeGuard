import time
import numpy as np
import config

class ConcentrationScorer:
    def __init__(self):
        self.score_history = []
        self.history_length = 30
        
        # Timers
        self.distracted_start_time = None
        self.eyes_closed_start_time = None
        
        # NEW: Timer for looking away (Gaze/Head)
        self.looking_away_start_time = None

    def _get_graded_score(self, value, low_limit, high_limit, buffer=0.05):
        # 1. Safe Zone
        if low_limit <= value <= high_limit:
            return 1.0
        # 2. Too Low
        if value < low_limit:
            diff = low_limit - value
            if diff > buffer: return 0.0
            return 1.0 - (diff / buffer)
        # 3. Too High
        if value > high_limit:
            diff = value - high_limit
            if diff > buffer: return 0.0
            return 1.0 - (diff / buffer)

    def get_score(self, ear, gaze_ratio, pitch, yaw):
        
        # --- 1. Blink Score ---
        if ear > config.EAR_THRESHOLD:
            score_blink = 1.0
            self.eyes_closed_start_time = None
        else:
            score_blink = 0.0
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = time.time()

        # --- 2. Calculate Raw Scores ---
        # Gaze Score (Using new relative thresholds)
        score_gaze = self._get_graded_score(
            gaze_ratio, 
            config.GAZE_LEFT_THRESHOLD, 
            config.GAZE_RIGHT_THRESHOLD, 
            buffer=0.1 # Increased buffer slightly for stability
        )
        
        # Head Pose Score
        score_pitch = self._get_graded_score(pitch, -config.PITCH_THRESHOLD, config.PITCH_THRESHOLD, buffer=5)
        score_yaw = self._get_graded_score(yaw, -config.YAW_THRESHOLD, config.YAW_THRESHOLD, buffer=5)
        score_pose = min(score_pitch, score_yaw)

        # --- 3. "Thinking" Grace Period (With Debounce) ---
        is_looking_away = (score_gaze < 1.0 or score_pose < 1.0)
        
        if is_looking_away:
            # Start timer if not started
            if self.looking_away_start_time is None:
                self.looking_away_start_time = time.time()
            
            # Calculate duration
            time_away = time.time() - self.looking_away_start_time
            
            # GRACE PERIOD: If < 1.5s, FORCE SCORE TO 100%
            if time_away < 1.5:
                score_gaze = 1.0
                score_pose = 1.0
            # If > 1.5s, the actual low scores will flow through
            
        else:
            # DEBOUNCE: Don't reset timer immediately!
            # Only reset if we are truly focused (e.g., gaze is perfect center)
            # This prevents the "jitter" from resetting the clock constantly.
            if self.looking_away_start_time:
                 # Check if we are "safely" back in bounds (not just on the edge)
                 if score_gaze == 1.0 and score_pose == 1.0:
                     self.looking_away_start_time = None

        # --- 4. Drowsiness Override ---
        if self.eyes_closed_start_time:
            time_closed = time.time() - self.eyes_closed_start_time
            if time_closed > 0.5:
                score_gaze = 0.0
                score_pose = 0.0

        # --- 5. Weighted Average ---
        raw_score = (score_gaze * config.WEIGHT_GAZE) + \
                    (score_pose * config.WEIGHT_POSE) + \
                    (score_blink * config.WEIGHT_BLINK)
        
        current_score = raw_score * 100
        
        # --- 6. Smoothing ---
        self.score_history.append(current_score)
        if len(self.score_history) > self.history_length:
            self.score_history.pop(0)
        smooth_score = int(np.mean(self.score_history))
        
        # --- 7. Status Logic ---
        explanation = "Focused"
        if smooth_score < 40:
            explanation = "Distracted!"
            if self.distracted_start_time is None:
                self.distracted_start_time = time.time()
        else:
            self.distracted_start_time = None
            
        distraction_duration = 0
        if self.distracted_start_time:
            distraction_duration = time.time() - self.distracted_start_time
            
        return smooth_score, distraction_duration, explanation