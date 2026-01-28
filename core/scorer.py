import time
import numpy as np
import config

class ConcentrationScorer:
    def __init__(self):
        self.score_history = []
        self.history_length = 30
        
        self.distracted_start_time = None
        self.eyes_closed_start_time = None
        self.looking_away_start_time = None

    def _get_graded_score(self, value, low_limit, high_limit, buffer=0.05):
        if low_limit <= value <= high_limit: return 1.0
        if value < low_limit:
            diff = low_limit - value
            if diff > buffer: return 0.0
            return 1.0 - (diff / buffer)
        if value > high_limit:
            diff = value - high_limit
            if diff > buffer: return 0.0
            return 1.0 - (diff / buffer)

    def get_score(self, ear, gaze_ratio, pitch, yaw, phone_detected=False):
        
        # --- Blink Score ---
        if ear > config.EAR_THRESHOLD:
            score_blink = 1.0
            self.eyes_closed_start_time = None
        else:
            score_blink = 0.0
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = time.time()

        # --- Calculate of Raw Scores ---
        # Gaze (Buffered)
        score_gaze = self._get_graded_score(gaze_ratio, config.GAZE_LEFT_THRESHOLD, config.GAZE_RIGHT_THRESHOLD, buffer=0.1)
        
        # Head Pose (Buffered)
        score_yaw = self._get_graded_score(yaw, -config.YAW_THRESHOLD, config.YAW_THRESHOLD, buffer=5)
        score_pitch = self._get_graded_score(pitch, -config.PITCH_THRESHOLD, config.PITCH_THRESHOLD, buffer=5)
        
        # Combined Head Score
        score_pose = min(score_pitch, score_yaw)
        if score_pose < 0.6:
            score_gaze = min(score_gaze, score_pose)

        # ---Distraction Logic ---
        explanation = "" 
        # PHONE (Instant Fail)
        if phone_detected:
            score_gaze = 0.0
            score_pose = 0.0
            self.score_history = [] 
            explanation = "Distracted (Phone)" 
            self.looking_away_start_time = None
            
        # LOOKING AWAY (Buffered)
        else:
            is_looking_away = (score_gaze < 1.0 or score_pose < 1.0)
            
            if is_looking_away:
                if self.looking_away_start_time is None:
                    self.looking_away_start_time = time.time()
                
                time_away = time.time() - self.looking_away_start_time
                
                if time_away < 1.5:
                    score_gaze = 1.0
                    score_pose = 1.0
            else:
                if self.looking_away_start_time:
                    if score_gaze == 1.0 and score_pose == 1.0:
                        self.looking_away_start_time = None

        # ---Drowsiness Override ---
        if self.eyes_closed_start_time:
            time_closed = time.time() - self.eyes_closed_start_time
            if time_closed > 0.5:
                score_gaze = 0.0
                score_pose = 0.0
                explanation = "Drowsy!"

        # ---Weighted Average ---
        raw_score = (score_gaze * config.WEIGHT_GAZE) + \
                    (score_pose * config.WEIGHT_POSE) + \
                    (score_blink * config.WEIGHT_BLINK)
        
        current_score = int(raw_score * 100)
        
        # ---Smoothing ---
        self.score_history.append(current_score)
        if len(self.score_history) > self.history_length:
            self.score_history.pop(0)
        smooth_score = int(np.mean(self.score_history))
        
        # ---Status Logic ---
        if explanation == "":
            explanation = "Focused"
            if smooth_score < 40:
                explanation = "Distracted!"
        
        # Timer logic
        if smooth_score < 40 or phone_detected:
            if self.distracted_start_time is None:
                self.distracted_start_time = time.time()
        else:
            self.distracted_start_time = None
            
        distraction_duration = 0
        if self.distracted_start_time:
            distraction_duration = time.time() - self.distracted_start_time
            
        return smooth_score, distraction_duration, explanation