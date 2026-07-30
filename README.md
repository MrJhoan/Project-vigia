# PROJECT VIGIA — Final Detection Prototype V3.0

This branch contains the **final and most complete version of the original PROJECT VIGIA detection prototype**, independently designed and developed by **Jhoan García**.

The prototype processes recorded videos or a live camera feed, analyzes frames with computer-vision models, identifies visual patterns associated with potentially criminal events, marks the people and objects involved, and activates an audible alarm when an event is confirmed.

> This is an experimental decision-support prototype. Its detections are not legal judgments and must always be reviewed by a person before any response or escalation.

## Why This Version Matters

Version 3.0 brings together the project's complete detection-and-alert workflow:

```text
Video or camera input
        ↓
Real-time frame processing
        ↓
Pose, object, and face analysis
        ↓
Temporal event validation
        ↓
Visual annotations + audible alarm
        ↓
WebSocket alert + test notification
```

Unlike a detector that evaluates each image in isolation, this prototype keeps information from multiple frames to reduce premature alerts and evaluate how movements evolve over time.

## Main Features

- Processes the test videos in `videos/` automatically
- Uses the computer's camera when no compatible video files are available
- Runs video playback and AI analysis in separate threads
- Analyzes one of every four frames to improve playback fluidity
- Detects people and estimates their body poses with YOLOv8 Pose
- Detects relevant objects with YOLOv8
- Uses OpenCV face detection as an additional motion reference
- Evaluates arm-pointing, hand-to-face movement, theft-related gestures, objects near a hand, people on the ground, and group-attack patterns
- Assigns visual roles such as aggressor, victim, and suspect
- Draws skeletons, bounding boxes, event labels, and system status directly on the video
- Activates an MP3 alarm when an aggressive event is confirmed
- Uses a system-beep fallback when the MP3 alarm cannot be played
- Applies warm-up frames, temporal confirmation, event locking, and alert cooldowns to reduce unstable detections
- Sends alert data and an annotated frame through a WebSocket endpoint
- Includes a simulated Twilio notification flow for local testing

## Detection Scenarios

The included test set was used to refine the prototype for several visual scenarios:

- A person pointing an arm in a weapon-related context
- Knife detection
- Rapid hand movement toward a person's face
- Theft-related body and hand gestures
- Movement of a small object near a hand
- A person lying on the ground
- Multiple people involved in a possible group attack

These are heuristic and model-assisted detections. Real-world performance can vary according to camera angle, lighting, occlusion, video quality, and the behavior shown in the scene.

## Technologies

- Python
- Ultralytics YOLOv8
- OpenCV
- NumPy
- FastAPI
- WebSockets
- Threading
- Playsound
- Twilio SDK in test mode

The prototype loads:

- `yolov8n-pose.pt` for pose estimation
- `yolov8n.pt` for object detection
- OpenCV's frontal-face Haar cascade

The YOLO model files are downloaded automatically by Ultralytics the first time they are needed.

## Project Structure

```text
Project-vigia/
├── main.py                 # FastAPI app, video playback, AI worker, and alerts
├── detector.py             # Detection logic and temporal event validation
├── alarm.py                # MP3 alarm and system-beep fallback
├── twilio_sender.py        # Simulated external notification flow
├── requirements.txt        # Python dependencies
├── IPhone Alarm Ringtone.mp3
└── videos/                 # Test videos
```

## Installation

Python 3.11 is recommended.

1. Clone the repository and select this branch:

   ```bash
   git clone --branch prototipo-V3.0 --single-branch https://github.com/MrJhoan/Project-vigia.git
   cd Project-vigia
   ```

2. Create and activate a virtual environment:

   **Windows**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS or Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Prototype

Start the FastAPI application from the repository root:

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

At startup, the application:

1. Finds supported files inside `videos/`.
2. Opens the first video, or the default camera if the folder is empty.
3. Starts the video and AI-processing threads.
4. Displays an OpenCV window with detections and system status.
5. Activates the alarm and dispatches an alert when an event is confirmed.

## Video Controls

Click the OpenCV video window before using the keyboard controls.

| Key | Action |
|---|---|
| `Space` | Pause or resume |
| `F` | Play at 2× speed |
| `S` | Play at 0.5× speed |
| `N` | Return to normal speed |
| `,` | Go back 10 seconds |
| `.` | Go forward 10 seconds |
| `Q` | Skip to the next video |

## Alert Integration

The application exposes a WebSocket endpoint at:

```text
ws://127.0.0.1:8000/ws/alerts
```

Each dispatched message contains the detected event, an annotated frame encoded in Base64, and a countdown value.

`twilio_sender.py` currently operates in **test mode** and prints notification information to the console. It does not contact the police or send real SMS messages unless a developer deliberately configures and enables the commented Twilio integration.

## What Was Improved in V3.0

- Prevented repeated analysis of the same frame
- Improved separation between video playback and AI processing
- Added cleaner state resets when switching videos
- Reduced premature alarms in the jewelry-store test scenarios
- Required stronger contextual evidence before confirming certain aggressive movements
- Improved temporal validation for arm pointing, hand-to-face movement, theft gestures, and group attacks
- Preserved stable aggressor and victim roles across consecutive frames
- Improved alarm continuity while avoiding repeated alert dispatches

## Authorship

The original idea, detection logic, real-time video-analysis workflow, event-validation rules, audible-alarm system, integration between the modules, testing, and iterative refinement of this prototype were completed by **Jhoan García**.

The prototype later served as the technical foundation and proof of concept for the broader [collaborative PROJECT VIGIA platform](https://github.com/MrJhoan/PROJECT_VIGIA).

## Responsible Use and Limitations

This repository is an academic and experimental prototype, not a production-ready surveillance or emergency-response system. It can produce false positives and false negatives. It must not be used for autonomous accusations, profiling, or decisions affecting a person's rights.

Any real deployment would require additional model validation, representative datasets, privacy and data-retention controls, security hardening, human oversight, regulatory review, and formal coordination with the relevant authorities.

