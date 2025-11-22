import time
from collections import deque
from body_part_angle import BodyPartAngle

def _safe(a):
    return None if a is None else float(a)

class TypeOfExercise(BodyPartAngle):
    """
    HIGH SENSITIVITY MODE:
    - Counts every rep immediately (no stability delay).
    - Counts partial reps (relaxed thresholds).
    - Counts regardless of posture/form.
    """

    SMOOTH_WINDOW = 3           # Low smoothing for fast response
    STABLE_FRAMES_REQUIRED = 1  # INSTANT TRIGGER (Changed from 3)
    MIN_REP_INTERVAL = 0.15     # Allows very fast reps

    def __init__(self, landmarks=None):
        super().__init__(landmarks)
        self.landmarks = landmarks
        self._buffers = {
            "left_elbow": deque(maxlen=self.SMOOTH_WINDOW),
            "right_elbow": deque(maxlen=self.SMOOTH_WINDOW),
            "left_knee": deque(maxlen=self.SMOOTH_WINDOW),
            "right_knee": deque(maxlen=self.SMOOTH_WINDOW),
            "abdomen": deque(maxlen=self.SMOOTH_WINDOW),
            "neck": deque(maxlen=self.SMOOTH_WINDOW),
        }
        self._smoothed = {}
        self._last_rep_time = {"push": 0.0, "squat": 0.0, "sit": 0.0, "pull": 0.0}

    def update_landmarks(self, landmarks):
        self.landmarks = landmarks
        try:
            a = self.angle_of_the_left_arm(); 
            if a is not None: self._buffers["left_elbow"].append(a)
        except: pass
        try:
            a = self.angle_of_the_right_arm(); 
            if a is not None: self._buffers["right_elbow"].append(a)
        except: pass
        try:
            a = self.angle_of_the_left_leg(); 
            if a is not None: self._buffers["left_knee"].append(a)
        except: pass
        try:
            a = self.angle_of_the_right_leg(); 
            if a is not None: self._buffers["right_knee"].append(a)
        except: pass
        try:
            a = self.angle_of_the_abdomen(); 
            if a is not None: self._buffers["abdomen"].append(a)
        except: pass
        try:
            a = self.angle_of_the_neck();
            if a is not None: self._buffers["neck"].append(a)
        except: pass

        for k, dq in self._buffers.items():
            self._smoothed[k] = (sum(dq) / len(dq)) if len(dq) > 0 else None

    def get_smoothed_angles(self):
        return dict(self._smoothed)

    def _can_count_rep(self, key):
        now = time.time()
        if now - self._last_rep_time.get(key, 0.0) >= self.MIN_REP_INTERVAL:
            self._last_rep_time[key] = now
            return True
        return False

    # -------------------------
    # Posture heuristics (Only for visual "True/False" feedback now)
    # -------------------------
    def posture_correct_push(self):
        abdomen = _safe(self._smoothed.get("abdomen"))
        le = _safe(self._smoothed.get("left_elbow"))
        re = _safe(self._smoothed.get("right_elbow"))
        if abdomen is None: return True # Default to True if unsure
        if abdomen < 140: return False
        if le is not None and re is not None and abs(le - re) > 30: return False
        return True

    def posture_correct_squat(self):
        lk = _safe(self._smoothed.get("left_knee"))
        rk = _safe(self._smoothed.get("right_knee"))
        if lk is None and rk is None: return True
        avg = lk if rk is None else (rk if lk is None else (lk + rk) / 2.0)
        if avg >= 90: return True
        return False

    def posture_correct_sit(self):
        abdomen = _safe(self._smoothed.get("abdomen"))
        neck = _safe(self._smoothed.get("neck"))
        if abdomen is None: return True
        if abdomen < 100: return False
        return True

    def posture_correct_pull(self):
        abdomen = _safe(self._smoothed.get("abdomen"))
        if abdomen is None: return True
        if abdomen < 100: return False
        return True

    def _progress_from_angle(self, angle, down_thresh, up_thresh, invert=False):
        if angle is None: return 0.0
        if not invert:
            if angle <= down_thresh: return 0.0
            if angle >= up_thresh: return 1.0
            return (angle - down_thresh) / (up_thresh - down_thresh)
        else:
            if angle >= down_thresh: return 0.0
            if angle <= up_thresh: return 1.0
            return (down_thresh - angle) / (down_thresh - up_thresh)

    # -------------------------
    # Exercise implementations
    # -------------------------
    def push_up(self, counter, stage):
        le = self._smoothed.get("left_elbow")
        re = self._smoothed.get("right_elbow")
        if le is None and re is None: return [counter, stage, False, 0.0]
        avg = le if re is None else (re if le is None else (le + re) / 2.0)

        # Relaxed Thresholds: Easier to count
        DOWN_THRESH = 100.0 
        UP_THRESH = 150.0
        key = "push"

        if stage is None:
            stage = "up" if avg > UP_THRESH else "down"

        if stage == "up":
            if avg < DOWN_THRESH:
                stage = "down"
        else: # stage is down
            if avg > UP_THRESH:
                # ALWAYS COUNT
                if self._can_count_rep(key):
                    counter += 1
                stage = "up"

        posture_bool = self.posture_correct_push()
        progress = self._progress_from_angle(avg, DOWN_THRESH, UP_THRESH, invert=False)
        return [counter, stage, posture_bool, progress]

    def pull_up(self, counter, stage):
        le = self._smoothed.get("left_elbow")
        re = self._smoothed.get("right_elbow")
        if le is None and re is None: return [counter, stage, False, 0.0]
        avg = le if re is None else (re if le is None else (le + re) / 2.0)

        # Relaxed Thresholds
        DOWN_THRESH = 145.0
        UP_THRESH = 95.0
        key = "pull"

        if stage is None:
            stage = "down" if avg > DOWN_THRESH else "up"

        if stage == "down":
            if avg < UP_THRESH:
                # ALWAYS COUNT
                if self._can_count_rep(key):
                    counter += 1
                stage = "up"
        else: # stage is up
            if avg > DOWN_THRESH:
                stage = "down"

        posture_bool = self.posture_correct_pull()
        progress = self._progress_from_angle(avg, DOWN_THRESH, UP_THRESH, invert=True)
        return [counter, stage, posture_bool, progress]

    def squat(self, counter, stage):
        lk = self._smoothed.get("left_knee")
        rk = self._smoothed.get("right_knee")
        if lk is None and rk is None: return [counter, stage, False, 0.0]
        avg = lk if rk is None else (rk if lk is None else (lk + rk) / 2.0)

        # Relaxed Thresholds: Half squats will now count
        DOWN_THRESH = 100.0  
        UP_THRESH = 150.0    
        key = "squat"

        if stage is None:
            stage = "up" if avg > UP_THRESH else "down"

        if stage == "up":
            if avg < DOWN_THRESH:
                stage = "down"
        else: # stage is down
            if avg > UP_THRESH:
                # ALWAYS COUNT
                if self._can_count_rep(key):
                    counter += 1
                stage = "up"

        posture_bool = self.posture_correct_squat()
        progress = self._progress_from_angle(avg, DOWN_THRESH, UP_THRESH, invert=False)
        return [counter, stage, posture_bool, progress]

    def sit_up(self, counter, stage):
        a = self._smoothed.get("abdomen")
        if a is None: return [counter, stage, False, 0.0]

        # Relaxed Thresholds: Partial crunches will count
        DOWN_THRESH = 80.0
        UP_THRESH = 100.0
        key = "sit"

        if stage is None:
            stage = "up" if a > UP_THRESH else "down"

        if stage == "up":
            if a < DOWN_THRESH:
                stage = "down"
        else: # stage is down
            if a > UP_THRESH:
                # ALWAYS COUNT
                if self._can_count_rep(key):
                    counter += 1
                stage = "up"

        posture_bool = self.posture_correct_sit()
        progress = self._progress_from_angle(a, DOWN_THRESH, UP_THRESH, invert=False)
        return [counter, stage, posture_bool, progress]

    def calculate_exercise(self, exercise_type, counter, stage):
        et = exercise_type.lower()
        if et == "push-up":
            return self.push_up(counter, stage)
        elif et == "pull-up":
            return self.pull_up(counter, stage)
        elif et == "squat":
            return self.squat(counter, stage)
        elif et == "sit-up":
            return self.sit_up(counter, stage)
        else:
            return [counter, stage, True, 0.0]