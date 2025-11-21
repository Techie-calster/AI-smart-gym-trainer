
import streamlit as st
import cv2
import os
import time

from engine import start_engine

VIDEO_DIR = "Exercise Videos"

def main():
    st.set_page_config(
        page_title="PostuRight – AI Trainer",
        layout="wide"
    )

    if "run" not in st.session_state:
        st.session_state.run = False

    if "last_report" not in st.session_state:
        st.session_state.last_report = None

    if "countdown_done" not in st.session_state:
        st.session_state.countdown_done = False

    # Sidebar
    st.sidebar.image("Brand_Logo.jpg", use_container_width=True)
    st.sidebar.header("Settings")

    exercise = st.sidebar.selectbox(
        "Select Exercise",
        ["squat", "push-up", "pull-up", "sit-up"]
    )

    source = st.sidebar.radio(
        "Select Video Source",
        ["Live Webcam", "Pre-recorded Video"]
    )

    video_path = None

    if source == "Pre-recorded Video":
        video_name = st.sidebar.text_input(
            "Enter video name (ex: squat_7.mp4)"
        )

        if video_name:
            potential = os.path.join(VIDEO_DIR, video_name)
            if os.path.exists(potential):
                video_path = potential
            else:
                video_path = video_name

    if st.sidebar.button("Start / Restart"):
        st.session_state.run = True
        st.session_state.countdown_done = False

    if st.sidebar.button("Stop"):
        st.session_state.run = False

    col1, col2, col3, col4 = st.columns(4)

    kpi_reps = col1.empty()
    kpi_stage = col2.empty()
    kpi_posture = col3.empty()
    kpi_fps = col4.empty()

    stframe = st.empty()

    # ---------------------------------------
    # DEFAULT SCREEN BEFORE START
    # ---------------------------------------
    if not st.session_state.run:
        st.markdown(
            """
            <div style='display:flex; justify-content:center; align-items:center; height:70vh;'>
                <div style='text-align:center; max-width:500px;'>
                    <h1 style='color:#00f2c3;'>🏋️ PostuRight AI Trainer</h1>
                    <p style='color:gray; font-size:18px; margin-top:20px;'>
                        Select your exercise and video input<br>
                        then click <strong>Start / Restart</strong> to begin
                    </p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ---------------------------------------
    # WHEN START IS PRESSED
    # ---------------------------------------
    if st.session_state.run:

        if source == "Live Webcam":
            video_source = 0
        else:
            if not video_path:
                st.error("Please enter a valid video path")
                return
            video_source = video_path

        # ----------------- COUNTDOWN ----------------
        if not st.session_state.countdown_done:
            placeholder = st.empty()

            for i in [3, 2, 1]:
                placeholder.markdown(
                    f"""
                    <div style='display:flex; justify-content:center; align-items:center; height:70vh;'>
                        <h1 style='font-size:120px; color:#00f2c3;'>{i}</h1>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                time.sleep(1)

            placeholder.markdown(
                """
                <div style='display:flex; justify-content:center; align-items:center; height:70vh;'>
                    <h1 style='font-size:60px; color:#00f2c3;'>START!</h1>
                </div>
                """,
                unsafe_allow_html=True
            )
            time.sleep(1)

            st.session_state.countdown_done = True
            placeholder.empty()

        # ----------------- STREAMING ----------------
        def display_callback(frame, reps, stage, posture, progress, fps):
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            stframe.image(frame_rgb, channels="RGB", use_container_width=True)

            kpi_reps.metric("Reps", reps)
            kpi_stage.metric("Stage", stage)
            kpi_posture.metric("Posture", "Good" if posture else "Bad")
            kpi_fps.metric("FPS", fps)

        def stop_callback():
            return st.session_state.run

        report = start_engine(
            exercise,
            video_source,
            display_callback,
            stop_callback
        )

        st.session_state.last_report = report
        st.session_state.run = False
        st.session_state.countdown_done = False

    # ---------------------------------------
    # SHOW LAST REPORT
    # ---------------------------------------
    if st.session_state.last_report:

        st.markdown("## 📄 Last Workout Report")

        r = st.session_state.last_report

        st.write(f"**Exercise:** {r['exercise']}")
        st.write(f"**Total Reps:** {r['reps']}")
        st.write(f"**Duration:** {r['duration']} seconds")
        st.write(f"**Accuracy:** {r['accuracy']:.2f}%")

        try:
            with open(r["report_path"], "r") as f:
                content = f.read()

            st.download_button(
                "Download report",
                content,
                file_name=os.path.basename(r["report_path"]),
                mime="text/plain"
            )

        except:
            st.warning("Report file not found.")


if __name__ == "__main__":
    main()
