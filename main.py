import cv2
import os
import time
import csv
import math
import mediapipe as mp
import numpy as np
from datetime import datetime
from exercise_classifier import ExerciseClassifier
from types_of_exercise import TypeOfExercise
from utils import score_table

# --- CONFIG ---
MODEL_PATH = "slowfps_gru_mps.pt"
DS_PATH = "slowfps_dataset.npz"
CSV_LOG_PATH = "./logs/exercise_session_log.csv"
os.makedirs("./logs", exist_ok=True)
CONF_THRESH = 0.75
SHOW_ANGLES = True
SHOW_PROGRESS_BAR = True
CALIBRATE_SECONDS = 3     # quick standing calibration duration
FPS_SMOOTH = 0.9          # low-pass for FPS display

# --- Audio helper (optional) ---
try:
    import simpleaudio as sa
    def play_tone(frequency=880, duration_ms=120, volume=0.2):
        fs = 44100
        t = np.linspace(0, duration_ms / 1000, int(fs * duration_ms / 1000), False)
        wave = np.sin(frequency * t * 2 * np.pi) * volume
        audio = (wave * 32767).astype(np.int16)
        play_obj = sa.play_buffer(audio, 1, 2, fs)
        # do not block
    AUDIO_AVAILABLE = True
except Exception:
    AUDIO_AVAILABLE = False
    def play_tone(frequency=880, duration_ms=120, volume=0.2):
        # fallback: no audio available
        pass

def beep_on_rep():
    if AUDIO_AVAILABLE:
        play_tone(1000, 140, 0.25)

def beep_bad_posture():
    if AUDIO_AVAILABLE:
        play_tone(300, 180, 0.2)

# --- Helpers ---
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def fmt_ang(a):
    return f"{int(a)}°" if a is not None else "N/A"

def ensure_csv_header(path):
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "exercise", "rep_num", "confidence", "posture", "notes"])

def log_rep(exercise, rep_num, confidence, posture, notes=""):
    ensure_csv_header(CSV_LOG_PATH)
    ts = datetime.utcnow().isoformat()
    with open(CSV_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ts, exercise, rep_num, f"{confidence:.3f}", posture, notes])

# --- Init classifier and tracker ---
classifier = ExerciseClassifier(model_path=MODEL_PATH, ds_path=DS_PATH)

# Ask user which mode: ML auto or Manual (menu)
print("\n=== MODE SELECTION ===")
print("1) ML auto-detect exercise (uses your GRU model)")
print("2) Manual mode (pick exercise from menu)")
mode = input("Choose (1/2) [default 1]: ").strip() or "1"
USE_ML = (mode == "1")

# Calibration prompt (optional)
print("\nCalibration: stand upright in frame for ~{} seconds (or press Enter to skip)".format(CALIBRATE_SECONDS))
inp = input("Press Enter to start calibration or type 'skip' to skip: ").strip().lower()
do_calib = (inp != "skip")

baseline_abdomen = None
if do_calib:
    print("Starting calibration: remain standing and centered...")
    cap_test = cv2.VideoCapture(0)
    start_t = time.time()
    frames = []
    with mp_pose.Pose(min_detection_confidence=0.5) as pose_c:
        while time.time() - start_t < CALIBRATE_SECONDS:
            ret, f = cap_test.read()
            if not ret:
                break
            res = pose_c.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            if res.pose_landmarks:
                # compute abdomen angle as in body_part_angle
                # reuse landmark indices via mp.solutions if available
                try:
                    lm = res.pose_landmarks.landmark
                    r_sh = [lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                            lm[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    l_sh = [lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            lm[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    shoulder_avg = [(r_sh[0] + l_sh[0]) / 2, (r_sh[1] + l_sh[1]) / 2]
                    r_hip = [lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].x,
                             lm[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value].y]
                    l_hip = [lm[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].x,
                             lm[mp.solutions.pose.PoseLandmark.LEFT_HIP.value].y]
                    hip_avg = [(r_hip[0] + l_hip[0]) / 2, (r_hip[1] + l_hip[1]) / 2]
                    r_knee = [lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].x,
                              lm[mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value].y]
                    l_knee = [lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].x,
                              lm[mp.solutions.pose.PoseLandmark.LEFT_KNEE.value].y]
                    knee_avg = [(r_knee[0] + l_knee[0]) / 2, (r_knee[1] + l_knee[1]) / 2]
                    # Compute angle shoulder-hip-knee (using same formula as utils.calculate_angle)
                    def calc_angle(a, b, c):
                        a = np.array(a); b = np.array(b); c = np.array(c)
                        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                        angle = abs(radians * 180.0 / np.pi)
                        if angle > 180: angle = 360 - angle
                        return angle
                    abd = calc_angle(shoulder_avg, hip_avg, knee_avg)
                    frames.append(abd)
                except Exception:
                    pass
    cap_test.release()
    if len(frames) > 0:
        baseline_abdomen = float(np.mean(frames))
        print(f"Calibration done — baseline abdomen angle ≈ {baseline_abdomen:.1f}°")
    else:
        print("Calibration failed (no landmarks). Proceeding without baseline.")

# Video selection (webcam or file)
print("\n====== VIDEO SOURCE SELECTION ======")
print("1. Use Webcam")
print("2. Use Pre-recorded Video")
choice = input("Select (1/2): ").strip()
if choice == "2":
    path = input("Enter full video file path: ").strip()
    if not os.path.exists(path):
        print("File not found. Exiting."); raise SystemExit
    cap = cv2.VideoCapture(path)
else:
    cap = cv2.VideoCapture(0)

# If manual mode, show exercise menu
manual_ex = None
if not USE_ML:
    print("\nManual Mode: choose exercise")
    print("1. squat\n2. push-up\n3. pull-up\n4. sit-up")
    ch = input("Select (1-4): ").strip()
    mapping = {"1":"squat","2":"push-up","3":"pull-up","4":"sit-up"}
    manual_ex = mapping.get(ch, "squat")
    print("Manual exercise selected:", manual_ex)

# instantiate tracker
tracker = TypeOfExercise(None)
counter = 0
stage = None
posture = False
progress = 0.0
predicted_exercise = "waiting"
conf = 0.0

# fps smoothing
fps = 0.0
last_time = time.time()

# ensure csv header
ensure_csv_header(CSV_LOG_PATH)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = pose.process(frame_rgb)
        frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # 1) predict exercise (if ML mode)
        if USE_ML:
            label, conf = classifier.predict(frame)
            if conf >= CONF_THRESH:
                predicted_exercise = label
            else:
                predicted_exercise = "waiting"
        else:
            predicted_exercise = manual_ex
            conf = 1.0

        # 2) update tracker smoothing buffers with landmarks if present
        if res.pose_landmarks:
            tracker.update_landmarks(res.pose_landmarks.landmark)

        # 3) compute rep & posture only when we have landmarks & an exercise
        if res.pose_landmarks and predicted_exercise != "waiting":
            prev_counter = counter
            counter, stage, posture, progress = tracker.calculate_exercise(predicted_exercise, counter, stage)
            # if a rep incremented, log and beep
            if counter != prev_counter:
                # log rep
                log_rep(predicted_exercise, counter, conf, posture)
                # audio cue
                beep_on_rep()
        else:
            # show waiting state
            if predicted_exercise == "waiting":
                posture = False

        # 4) skeleton color (green good, red bad)
        if res.pose_landmarks:
            color = (0,255,0) if posture else (0,0,255)
            mp_drawing.draw_landmarks(
                frame,
                res.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=color, thickness=3, circle_radius=3)
            )

        # 5) show angles optionally (smoothed)
        if SHOW_ANGLES:
            sm = tracker.get_smoothed_angles()
            y0 = 30
            et = predicted_exercise.lower() if isinstance(predicted_exercise, str) else "unknown"
            if et == "squat":
                left_k = sm.get("left_knee"); right_k = sm.get("right_knee")
                ltxt = f"Knee L: {int(left_k) if left_k else 'N/A'}"
                rtxt = f"Knee R: {int(right_k) if right_k else 'N/A'}"
                cv2.putText(frame, ltxt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, rtxt, (10, y0+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            elif et in ("push-up","pull-up"):
                le = sm.get("left_elbow"); re = sm.get("right_elbow")
                cv2.putText(frame, f"Elbow L: {int(le) if le else 'N/A'}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(frame, f"Elbow R: {int(re) if re else 'N/A'}", (10, y0+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            elif et == "sit-up":
                a = sm.get("abdomen")
                cv2.putText(frame, f"Torso: {int(a) if a else 'N/A'}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # 6) show score table and additional info
        posture_text = "Good" if posture else "Bad"
        frame = score_table(predicted_exercise, frame, counter, posture_text)
        cv2.putText(frame, f"Conf: {conf:.2f}", (10, frame.shape[0]-80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        # progress bar (left side)
        if SHOW_PROGRESS_BAR:
            bar_w = 24; bar_h = 220; margin = 12
            x0 = margin; y0_bar = int((frame.shape[0]-bar_h)/2); x1 = x0 + bar_w; y1 = y0_bar + bar_h
            cv2.rectangle(frame, (x0,y0_bar),(x1,y1),(200,200,200),2)
            fill_h = int(bar_h * progress)
            fill_y0 = y1 - fill_h
            fill_color = (0,255,0) if posture else (0,0,255)
            if fill_h > 0:
                cv2.rectangle(frame, (x0+2, fill_y0),(x1-2,y1-2), fill_color, -1)
            cv2.putText(frame, f"{int(progress*100)}%", (x1+6, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

        # 7) FPS counter (smoothed)
        t1 = time.time()
        frame_dt = t1 - last_time
        last_time = t1
        if frame_dt > 0:
            fps_inst = 1.0/frame_dt
            fps = FPS_SMOOTH*fps + (1-FPS_SMOOTH)*fps_inst
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

        # predicted label top-right
        cv2.putText(frame, f"Pred: {predicted_exercise}", (frame.shape[1]-260, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,0), 2)

        cv2.imshow("ML Exercise Tracker (merged)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # manual beep for testing: press 'b' to beep
        if key == ord('b'):
            beep_on_rep()

cap.release()
cv2.destroyAllWindows()

