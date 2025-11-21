# AI Smart Gym Trainer

A Python-based tool that analyzes exercise videos to provide posture tracking and basic feedback using computer vision. It uses MediaPipe for pose detection and OpenCV for video processing to help users monitor form during workouts.

## Features
- Pose and landmark detection via MediaPipe
- Process local video files or webcam input
- Simple rep/posture tracking examples
- Lightweight and easy to run locally

## Prerequisites
- Python 3.10
- Git (optional)
- Recommended: create and use a virtual environment

## Installation
1. Clone the repository (optional):
   git clone https://github.com/Techie-calster/AI-smart-gym-trainer.git
   cd AI-smart-gym-trainer

2. Create and activate a virtual environment (recommended):
   macOS / Linux:
   python3.10 -m venv venv
   source venv/bin/activate

   Windows (PowerShell):
   python -m venv venv
   venv\Scripts\Activate.ps1

3. Install required packages:
   pip install mediapipe==0.8.9.1 numpy==1.21.6 opencv-python==4.5.5.64 pandas==1.3.5

4. Fix protobuf compatibility (important for MediaPipe):
   pip uninstall protobuf -y
   pip install protobuf==3.20.3

## Running
- Run the main script:
  python main.py
- Follow prompts to provide a path to a local video file or specify webcam input (e.g., 0).
- Example flags if implemented:
  python main.py --video path/to/video.mp4
  python main.py --webcam 0

## Project structure
- main.py — entry point for processing video/webcam input
- requirements.txt — pinned dependencies (not always present)
- utils/ — helper modules and utilities
- models/ — model/config files (if any)

## Notes & Troubleshooting
- MediaPipe can be sensitive to newer protobuf versions; installing protobuf==3.20.3 avoids common runtime errors.
- If you encounter OpenCV or MediaPipe errors, try reinstalling dependencies inside a fresh virtual environment.
- If your camera or video feed doesn't open, verify permissions and the correct device index.

## Contributing
Contributions are welcome. Please open issues or PRs with clear descriptions and small, focused changes.

## License
Specify a license (for example, MIT) in a LICENSE file. Let me know which license you prefer and I can add it.

## Contact
Repository owner: Techie-calster