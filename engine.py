import cv2
import time
import os
import csv
from datetime import datetime
import mediapipe as mp

from types_of_exercise import TypeOfExercise
from utils import score_table


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

REPORT_DIR = "reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def fmt_ang(a):
    return f"{int(a)}°" if a is not None else "N/A"


def start_engine(
    exercise_type,
    video_source,
    display_callback=None,
    stop_callback=None
):
    """
    Core fitness tracking engine.
    Can be used by:
    - Streamlit
    - Terminal
    - Flask / FastAPI
    """

    cap = cv2.VideoCapture(video_source)
    cap.set(3, 800)
    cap.set(4, 480)

    tracker = TypeOfExercise(None)
    counter = 0
    stage = None
    posture = False
    progress = 0.0

    start_time = time.time()
    good_frames = 0
    bad_frames = 0
    prev_time = 0

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        while cap.isOpened():

            # Stop condition from Streamlit
            if stop_callback and stop_callback() is False:
                break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (800, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            results = pose.process(rgb)

            rgb.flags.writeable = True
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            landmarks = None
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

            if landmarks is not None:
                tracker.update_landmarks(landmarks)

            counter, stage, posture, progress = tracker.calculate_exercise(
                exercise_type, counter, stage
            )

            if posture:
                good_frames += 1
            else:
                bad_frames += 1

            smoothed = tracker.get_smoothed_angles()
            debug = []

            if exercise_type == "squat":
                debug.append(f"Knee L: {fmt_ang(smoothed.get('left_knee'))}")
                debug.append(f"Knee R: {fmt_ang(smoothed.get('right_knee'))}")

            elif exercise_type in ("push-up", "pull-up"):
                debug.append(f"Elbow L: {fmt_ang(smoothed.get('left_elbow'))}")
                debug.append(f"Elbow R: {fmt_ang(smoothed.get('right_elbow'))}")

            elif exercise_type == "sit-up":
                debug.append(f"Torso: {fmt_ang(smoothed.get('abdomen'))}")

            posture_text = "Good" if posture else "Bad"
            frame = score_table(exercise_type, frame, counter, posture_text)

            color = (0, 255, 0) if posture else (0, 0, 255)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(
                        color=(255, 255, 255),
                        thickness=2,
                        circle_radius=2
                    ),
                    mp_drawing.DrawingSpec(
                        color=color,
                        thickness=3,
                        circle_radius=3
                    ),
                )

            for i, txt in enumerate(debug):
                cv2.putText(frame, txt, (10, 30 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2)

            cv2.putText(frame, f"Stage: {stage}", (10, 440),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.putText(frame, f"Reps: {counter}", (10, 470),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 255), 2)

            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if prev_time else 0
            prev_time = curr_time

            if display_callback:
                display_callback(
                    frame,
                    counter,
                    stage,
                    posture,
                    progress,
                    fps
                )

    cap.release()

    # ---------------- REPORT ----------------
    end_time = time.time()
    duration = int(end_time - start_time)
    total_frames = good_frames + bad_frames
    accuracy = (good_frames / total_frames) * 100 if total_frames else 0

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    report_name = f"{exercise_type}_{timestamp}.txt"
    report_path = os.path.join(REPORT_DIR, report_name)

    with open(report_path, "w") as f:
        f.write("------ PostuRight AI Fitness Report ------\n\n")
        f.write(f"Exercise        : {exercise_type}\n")
        f.write(f"Total Reps      : {counter}\n")
        f.write(f"Duration        : {duration} seconds\n")
        f.write(f"Good Frames     : {good_frames}\n")
        f.write(f"Bad Frames      : {bad_frames}\n")
        f.write(f"Accuracy        : {accuracy:.2f}%\n")
        f.write(f"Date            : {datetime.now()}\n")

    csv_path = os.path.join(REPORT_DIR, "history.csv")

    file_exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["Date", "Exercise", "Reps", "Duration(s)", "Accuracy(%)"]
            )

        writer.writerow([
            datetime.now(),
            exercise_type,
            counter,
            duration,
            round(accuracy, 2)
        ])

    return {
        "exercise": exercise_type,
        "reps": counter,
        "duration": duration,
        "accuracy": accuracy,
        "report_path": report_path
    }
