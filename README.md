# Spot Person Follower — Visual Servoing with YOLOv8

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://docker.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6F61)](https://docs.ultralytics.com)
[![Spot SDK](https://img.shields.io/badge/Spot-Boston%20Dynamics-FFD700)](https://dev.bostondynamics.com)

We built an autonomous person-following system for the **Boston Dynamics Spot** robot. It uses a **ZED 2i stereo camera** for image capture, **YOLOv8** for real-time person detection, and a **centroid-based visual servoing controller** that turns bounding-box errors into velocity commands. The whole thing runs inside Docker with GPU acceleration — one command to build, one command to run.

The idea was simple: point Spot at a person, and have it follow them around — adjusting speed, turning, and even body pitch (for stairs) — all from a single camera feed. No depth estimation, no fancy planning. Just proportional control on three error signals and a state machine to handle what happens when the person disappears.

The entire codebase was implemented in **under 24 hours**. That's the perception pipeline, visual servoing controller, state machine, web streaming server, Docker infrastructure, and test suite — all in a day.

### Demo

<video src="https://github.com/user-attachments/assets/e9305f95-9c92-40be-8fdf-2696bce7b857" controls width="100%">
  Your browser does not support the video tag. <a href="media/Spot_follower_visual_servo_an.mp4">Download the demo video</a>.
</video>

---

## What It Does

Spot stands up, starts grabbing frames from the ZED 2i at 720p/30fps, and runs YOLOv8 on each frame to find people. When it locks onto someone, the visual servoing controller kicks in and computes three error signals:

- **Lateral error** — how far the person's bounding box center is from the image center horizontally → turns the robot
- **Distance error** — how different the bounding box area is from a target size → moves the robot forward/backward
- **Pitch error** — how far the person is from center vertically → tilts Spot's body up/down (useful on stairs)

These errors get run through proportional gains, clamped to safe velocity limits, smoothed with acceleration ramping, and sent to Spot at 10 Hz. There's also a Flask web server on port 5000 that streams the annotated camera feed so you can watch what Spot sees from your laptop.

```
ZED 2i Camera (1280×720 @ 30fps)
        │
  YOLOv8 Inference (GPU) ──► Bounding Box + Confidence
        │
  Behavior State Machine
        │
        ├── TRACKING → Visual Servoing → v_x, ω, pitch → Spot SDK
        ├── SEARCH   → Rotate in place (looking for person)
        └── STOPPED  → Idle (search timed out)
```

The state machine handles the transitions: if detection is lost for more than 2 seconds, Spot starts rotating to search. If it doesn't find anyone after 30 seconds, it gives up and stops. The moment it re-detects a person, it snaps back to tracking.

---

## How the Visual Servoing Works

The controller is basically three P-controllers running in parallel. Nothing fancy — no PID, no model predictive control — just proportional gains with some engineering to make it smooth:

1. **Compute errors** from the bounding box vs. image center and target area
2. **Apply deadband** — ignore small errors to prevent jitter
3. **Saturate velocities** — hard clamp to safety limits (0.5 m/s forward, 0.6 rad/s turning)
4. **Ramp acceleration** — limit how fast velocities can change between cycles
5. **Send commands** to Spot at 10 Hz

| Parameter | Value | What it does |
|---|---|---|
| `kp_linear` | 0.5 | Forward/backward aggressiveness |
| `kp_angular` | 0.5 | Turning aggressiveness |
| `kp_pitch` | 0.2 | Body tilt aggressiveness |
| `target_bbox_area` | 168750 px² | Following distance (bigger = closer) |
| `max_linear_velocity` | 0.5 m/s | Speed cap |
| `max_angular_velocity` | 0.6 rad/s | Turn rate cap |
| `confidence_threshold` | 0.8 | YOLO confidence filter |

All of these are configurable in `config/config.yaml` so you can tune behavior without rebuilding the Docker image.

---

## Project Structure

```
├── src/
│   ├── main.py                 # Entry point — control loop + Flask web server
│   ├── perception.py           # ZED 2i camera wrapper + YOLOv8 person detector
│   ├── visual_servoing.py      # Proportional control law (3-axis)
│   ├── state_machine.py        # TRACKING / SEARCH / STOPPED behavior FSM
│   ├── spot_controller.py      # Boston Dynamics Spot SDK wrapper
│   └── camera_viewer.py        # Standalone camera viewer utility
│
├── config/
│   └── config.yaml.example     # Copy to config.yaml and fill in Spot credentials
│
├── tests/
│   ├── test_detection.py       # Test YOLO with webcam
│   ├── test_camera.py          # Test Spot camera connection
│   └── test_mobility.py        # Test robot movement ( robot will move)
│
├── utils/
│   └── visualization.py        # Debug overlays and visualization
│
├── media/
│   └── Spot_follower_visual_servo.mp4
│
├── Dockerfile                  # Based on stereolabs/zed:4.2-runtime-cuda12.1-ubuntu22.04
├── docker-compose.yml          # 6 services: run, display, tests, shell
├── docker_run.sh               # CLI helper — ./docker_run.sh build | run | test-*
├── requirements.txt
└── setup_environment.sh
```

---

## Docker Setup

Everything runs in Docker. The image is based on `stereolabs/zed:4.2-runtime-cuda12.1-ubuntu22.04` which comes with the ZED SDK and CUDA pre-installed. On top of that we layer PyTorch, YOLOv8, the Boston Dynamics SDK, OpenCV, and Flask.

One gotcha we ran into: the ZED base image ships with `numpy 1.26.4`, and PyTorch/Ultralytics try to upgrade it to 2.x, which breaks the ZED Python bindings. The Dockerfile forces `numpy==1.26.4` after all other installs to fix this.

```bash
# Build
./docker_run.sh build

# Run
./docker_run.sh run

# Run with X11 display forwarding
./docker_run.sh run-display

# Check prerequisites (Docker, NVIDIA, GPU, network to Spot)
./docker_run.sh check
```

The `docker-compose.yml` defines six services — the main follower (headless and with display), three test containers (detection, camera, mobility), and an interactive shell for debugging.

---

## Running It

**You'll need:** A Boston Dynamics Spot robot, an NVIDIA GPU with (This was an external CPU attached to the robot in our case), Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), and optionally a ZED 2i camera (falls back to Spot's onboard cameras).

```bash
# Clone and configure
git clone https://github.com/vivekmattam02/spot.git
cd spot
cp config/config.yaml.example config/config.yaml
# Edit config.yaml with your Spot's IP, username, and password

# Build and run
./docker_run.sh build
./docker_run.sh run

# Watch the live feed
# Open http://<your-ip>:5000 in a browser
```

### Testing

You can test individual components without running the full system:

```bash
./docker_run.sh test-detection   # YOLO detection with your webcam
./docker_run.sh test-camera      # Spot camera connection
./docker_run.sh test-mobility    # Robot movement ( ROBOT WILL MOVE!)
./docker_run.sh shell            # Interactive shell for debugging
```

---

## Safety

We built in multiple safety layers because, well, Spot is expensive and heavy:

- **Hardware E-Stop** on the controller — always available
- **Software E-Stop** with keepalive timeout
- **Velocity saturation** — hard limits on speed and turn rate
- **Acceleration ramping** — smooth transitions, no jerky starts
- **Deadband** — ignores tiny errors so the robot doesn't twitch
- **Auto-stop** — immediately stops when person detection is lost
- **Search timeout** — stops rotating after 30 seconds if nobody is found

---

## Team

- **Vivekanada Swamy Mattam**
- **Rahul Reghunath**
- **Tarunkumar Palanivelan** 
- **Jotheesh Reddy K** 

