# Spot Person Follower

Autonomous person-following system for the Boston Dynamics Spot robot using visual servoing and YOLOv8.

## Overview

This system enables Spot to autonomously follow a person using:
- ZED 2i stereo camera for image capture
- YOLOv8 for real-time person detection
- Proportional control for smooth tracking
- Body pitch control for tracking on stairs

## Features

- Real-time person detection and tracking
- Visual servoing with three control signals (lateral, distance, pitch)
- Automatic search behavior when person is lost
- Live web stream for monitoring (Flask server on port 5000)
- Docker support with GPU acceleration

## Requirements

- Boston Dynamics Spot robot
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- ZED 2i camera (optional, falls back to Spot's cameras)

## Quick Start

1. Clone and setup:
```bash
git clone https://github.com/vivekmattam02/spot.git
cd spot
cp config/config.yaml.example config/config.yaml
# Edit config/config.yaml with your Spot credentials
```

2. Build Docker image:
```bash
./docker_run.sh build
```

3. Run:
```bash
./docker_run.sh run
```

4. View live feed at `http://<your-ip>:5000`

## Configuration

Copy `config/config.yaml.example` to `config/config.yaml` and set:
- `spot.hostname`: Your Spot's IP address
- `spot.username`: Your Spot username
- `spot.password`: Your Spot password

Key tuning parameters:
- `control.kp_linear`: Forward/backward responsiveness (default: 0.5)
- `control.kp_angular`: Turning responsiveness (default: 0.5)
- `control.kp_pitch`: Body tilt responsiveness (default: 0.2)
- `target.target_bbox_area`: Following distance (larger = closer)

## Project Structure

```
spot/
├── config/
│   └── config.yaml.example    # Configuration template
├── src/
│   ├── main.py                # Entry point, control loop, web server
│   ├── spot_controller.py     # Spot SDK interface
│   ├── perception.py          # ZED camera + YOLO detection
│   ├── visual_servoing.py     # Control law implementation
│   └── state_machine.py       # Behavior state management
├── tests/                     # Test scripts
├── utils/                     # Visualization utilities
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## How It Works

1. Camera captures frame (1280x720 @ 30fps)
2. YOLOv8 detects person, returns bounding box
3. Visual servoing computes three errors:
   - Lateral error: horizontal offset from image center
   - Distance error: difference from target bounding box area
   - Pitch error: vertical offset from image center
4. Proportional control converts errors to velocity commands
5. Commands sent to Spot at 10Hz

## Safety

- Hardware E-Stop always available
- Software E-Stop with keepalive
- Velocity limits enforced
- Smooth acceleration ramping
- Automatic stop on detection loss

## Testing

```bash
./docker_run.sh test-detection   # Test YOLO with webcam
./docker_run.sh test-camera      # Test Spot camera
./docker_run.sh test-mobility    # Test robot movement (robot will move!)
```

## License

MIT License
