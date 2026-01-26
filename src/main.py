#!/usr/bin/env python3
"""
Spot Person Follower - Main Control Loop
=========================================
This is the main entry point for the Spot person-following system.
It orchestrates all components: perception, state machine, visual servoing,
and robot control.

System Overview:
---------------
    Camera → YOLO Detection → State Machine → Visual Servoing → Spot Movement
                  ↓                              ↓
             Detection              Velocity Commands (v_x, omega)

Control Loop (10 Hz):
    1. Capture image from Spot's camera
    2. Run YOLO detection to find persons
    3. Update state machine (TRACKING/SEARCH)
    4. If TRACKING: compute visual servoing command
    5. If SEARCH: use slow rotation
    6. Send velocity command to Spot
    7. Visualize (optional)
    8. Sleep to maintain 10 Hz rate

Safety Features:
---------------
- Graceful shutdown on Ctrl+C (sit down, release lease)
- E-Stop integration
- Velocity limits enforced at multiple levels
- Detection timeout triggers search, then stop
- Keyboard interrupt → immediate stop

Usage:
------
    python src/main.py                    # Use default config
    python src/main.py --config custom.yaml  # Use custom config

Author: Generated for Spot Person Follower Project
"""

import sys
import os
import time
import signal
import logging
import argparse
import threading
from pathlib import Path
from typing import Optional

import yaml
import cv2
import numpy as np
from flask import Flask, Response, render_template_string

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

# Import project modules
from .spot_controller import SpotController
from .perception import PersonDetector, Detection, ZEDCamera
from .visual_servoing import VisualServoingController, ControlOutput
from .state_machine import BehaviorStateMachine, BehaviorState

# Try to import visualization (optional)
try:
    from .visualization import Visualizer
    HAS_VISUALIZATION = True
except ImportError:
    HAS_VISUALIZATION = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress Flask/werkzeug logging
logging.getLogger('werkzeug').setLevel(logging.WARNING)

# =============================================================================
# Web Server for Live Camera Feed
# =============================================================================
# Global variables for frame sharing between control loop and web server
_latest_frame = None
_frame_lock = threading.Lock()

# Flask app for web streaming
flask_app = Flask(__name__)

WEB_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Spot Person Follower - Live</title>
    <style>
        body { background: #1a1a2e; color: #eee; font-family: Arial; 
               display: flex; flex-direction: column; align-items: center; padding: 20px; }
        h1 { color: #00d4ff; }
        img { border: 3px solid #00d4ff; border-radius: 8px; max-width: 100%; }
        .status { margin-top: 15px; padding: 10px 20px; background: #2d2d44; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🤖 Spot Person Follower - Live Feed</h1>
    <div>Robot is actively following. Detection boxes shown in green.</div>
    <br>
    <img src="/video_feed" alt="Live Feed">
    <div class="status"><strong>Stream active</strong> - Refresh if frozen</div>
</body>
</html>
"""

@flask_app.route('/')
def web_index():
    return render_template_string(WEB_HTML)

@flask_app.route('/video_feed')
def web_video_feed():
    def generate():
        while True:
            with _frame_lock:
                if _latest_frame is not None:
                    _, buffer = cv2.imencode('.jpg', _latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def start_web_server(port=5000):
    """Start Flask web server in a background thread."""
    flask_app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)



class PersonFollower:
    """
    Main orchestrator for the Spot person-following system.

    This class coordinates all subsystems:
    - SpotController: Robot interface
    - PersonDetector: YOLO-based person detection
    - VisualServoingController: Proportional control law
    - BehaviorStateMachine: State management
    - Visualizer: Optional debug visualization

    Usage:
        follower = PersonFollower(config_path="config/config.yaml")
        follower.run()
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        """
        Initialize the person follower system.

        Args:
            config_path: Path to YAML configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        logger.info(f"Loaded configuration from {config_path}")

        # Extract config sections
        spot_config = self.config.get("spot", {})
        control_config = self.config.get("control", {})
        safety_config = self.config.get("safety", {})
        perception_config = self.config.get("perception", {})
        target_config = self.config.get("target", {})
        state_config = self.config.get("state_machine", {})
        debug_config = self.config.get("debug", {})

        # Initialize Spot controller (for movement only, not camera)
        self.spot = SpotController(
            hostname=spot_config.get("hostname", "192.168.80.3"),
            username=spot_config.get("username", "admin"),
            password=spot_config.get("password", ""),
            camera_source=spot_config.get("camera_source", "frontleft_fisheye_image")
        )

        # Initialize ZED camera (for image capture)
        # ZED camera is connected via USB to the onboard computer (Apollo)
        zed_config = self.config.get("zed", {})
        self.zed_camera = ZEDCamera(
            resolution=zed_config.get("resolution", "HD720"),
            fps=zed_config.get("fps", 30)
        )
        self.use_zed_camera = zed_config.get("enabled", True)

        # Initialize person detector
        self.detector = PersonDetector(
            model_path=perception_config.get("model_path", "yolov8s.pt"),
            confidence_threshold=perception_config.get("confidence_threshold", 0.3),
            use_gpu=perception_config.get("use_gpu", True)
        )

        # Initialize visual servoing controller
        self.servoing = VisualServoingController(
            kp_linear=control_config.get("kp_linear", 0.3),
            kp_angular=control_config.get("kp_angular", 0.5),
            target_bbox_area=target_config.get("target_bbox_area", 30000),
            image_width=target_config.get("image_width", 640),
            image_height=target_config.get("image_height", 480),
            max_linear_velocity=safety_config.get("max_linear_velocity", 0.5),
            max_angular_velocity=safety_config.get("max_angular_velocity", 0.6),
            deadband=control_config.get("deadband", 0.05),
            velocity_ramp_rate=safety_config.get("velocity_ramp_rate", 0.1),
            kp_pitch=control_config.get("kp_pitch", 0.3),
            max_pitch_up=control_config.get("max_pitch_up", 0.26),
            max_pitch_down=control_config.get("max_pitch_down", 0.17)
        )

        # Initialize behavior state machine
        self.state_machine = BehaviorStateMachine(
            detection_timeout=safety_config.get("detection_timeout", 2.0),
            search_angular_velocity=state_config.get("search_angular_velocity", 0.2),
            max_search_duration=state_config.get("max_search_duration", 30.0)
        )

        # Initialize visualizer (optional)
        self.visualizer: Optional[Visualizer] = None
        if HAS_VISUALIZATION and debug_config.get("show_visualization", False):
            try:
                self.visualizer = Visualizer(
                    window_name="Spot Person Follower",
                    image_width=target_config.get("image_width", 640),
                    image_height=target_config.get("image_height", 480)
                )
            except Exception as e:
                logger.warning(f"Could not initialize visualizer: {e}")

        # Control loop parameters
        self.control_rate_hz = control_config.get("control_rate_hz", 10)
        self.control_period = 1.0 / self.control_rate_hz

        # Debug settings
        self.verbose = debug_config.get("verbose_logging", True)
        self.save_frames = debug_config.get("save_frames", False)
        self.save_path = debug_config.get("save_frames_path", "./debug_frames/")

        # Running state
        self._running = False
        self._shutdown_requested = False

        # Statistics
        self._loop_count = 0
        self._start_time: Optional[float] = None

        logger.info("PersonFollower initialized successfully")

    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary
        """
        # Handle relative paths
        if not os.path.isabs(config_path):
            # Try relative to script location
            script_dir = Path(__file__).parent.parent
            config_path = script_dir / config_path

        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            logger.warning("Using default configuration")
            return {}

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config if config else {}

    def _setup_signal_handlers(self) -> None:
        """
        Set up signal handlers for graceful shutdown.

        Handles:
        - SIGINT (Ctrl+C)
        - SIGTERM
        """
        def signal_handler(signum, frame):
            logger.warning(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_requested = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def run(self) -> None:
        """
        Run the main person-following control loop.

        This is the main entry point that:
        1. Connects to Spot
        2. Makes Spot stand
        3. Runs the control loop
        4. Handles graceful shutdown

        The loop runs until interrupted (Ctrl+C) or an error occurs.
        """
        logger.info("=" * 60)
        logger.info("Starting Spot Person Follower")
        logger.info("=" * 60)

        # Set up signal handlers
        self._setup_signal_handlers()

        try:
            # Open ZED camera first (connected to Apollo via USB)
            if self.use_zed_camera:
                logger.info("Opening ZED camera...")
                if not self.zed_camera.open():
                    raise RuntimeError("Failed to open ZED camera")
                logger.info("ZED camera opened successfully")

            # Connect to Spot (for movement control only)
            logger.info("Connecting to Spot robot...")
            self.spot.connect()
            logger.info("Connected to Spot")

            # Stand up
            logger.info("Commanding Spot to stand...")
            self.spot.stand()
            logger.info("Spot is standing")

            # Give user time to position themselves
            logger.info("")
            logger.info("*" * 60)
            logger.info("READY! Position yourself in front of Spot.")
            logger.info("Person following will begin in 3 seconds...")
            logger.info("Press Ctrl+C to stop at any time.")
            logger.info("*" * 60)
            logger.info("")
            
            # Start web server in background thread for live camera feed
            logger.info("Starting web server for live camera feed...")
            logger.info("View live feed at: http://<apollo-ip>:5000")
            web_thread = threading.Thread(target=start_web_server, daemon=True)
            web_thread.start()
            
            time.sleep(3.0)

            # Run main control loop
            self._run_control_loop()

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        except Exception as e:
            logger.error(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Graceful shutdown
            self._shutdown()

    def _run_control_loop(self) -> None:
        """
        Main control loop implementation.

        Runs at ~10 Hz (configurable) and performs:
        1. Image capture
        2. Person detection
        3. State machine update
        4. Velocity command computation
        5. Command execution
        6. Visualization (optional)
        """
        self._running = True
        self._start_time = time.time()

        logger.info("Entering main control loop...")

        while self._running and not self._shutdown_requested:
            loop_start = time.time()
            self._loop_count += 1

            try:
                # ===========================================================
                # Step 1: Capture image from ZED camera (USB connected to Apollo)
                # ===========================================================
                if self.use_zed_camera:
                    image = self.zed_camera.get_frame()
                else:
                    image = self.spot.get_image()

                if image is None:
                    logger.warning("Failed to capture image")
                    time.sleep(self.control_period)
                    continue

                # Flip image if using ZED (camera is mounted upside down)
                if self.use_zed_camera:
                    import cv2
                    image = cv2.flip(image, -1)  # -1 = rotate 180 degrees

                # ===========================================================
                # Step 2: Run person detection
                # ===========================================================
                detection = self.detector.detect(image)

                # ===========================================================
                # Step 3: Update state machine
                # ===========================================================
                state_output = self.state_machine.update(detection)

                # ===========================================================
                # Step 4: Compute velocity command based on state
                # ===========================================================
                v_x = 0.0
                v_y = 0.0
                omega = 0.0
                pitch = 0.0  # Body pitch for tracking elevated targets

                if state_output.state == BehaviorState.TRACKING:
                    if state_output.should_servo and detection is not None:
                        # Use visual servoing controller
                        control = self.servoing.compute(detection)
                        v_x = control.v_x
                        omega = control.omega
                        pitch = control.pitch  # Get pitch for tracking elevated people
                    else:
                        # Lost detection but within timeout - stop
                        v_x = 0.0
                        omega = 0.0
                        pitch = 0.0

                elif state_output.state == BehaviorState.SEARCH:
                    # Rotate to search for person
                    v_x = 0.0
                    omega = state_output.search_omega
                    pitch = 0.0  # Level body while searching

                elif state_output.state == BehaviorState.STOPPED:
                    # Stay stopped
                    v_x = 0.0
                    omega = 0.0
                    pitch = 0.0

                # ===========================================================
                # Step 5: Send velocity command to Spot
                # ===========================================================
                if self.spot.is_standing:
                    self.spot.move(v_x, v_y, omega, pitch=pitch)

                # ===========================================================
                # Step 6: Update web feed with detection overlay
                # ===========================================================
                global _latest_frame, _frame_lock
                display_frame = image.copy()
                
                # Draw detection box if person detected
                if detection is not None:
                    x1, y1, x2, y2 = [int(v) for v in detection.bbox]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"Person {detection.confidence:.2f}"
                    cv2.putText(display_frame, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.circle(display_frame, (int(detection.center_x), int(detection.center_y)), 
                              5, (0, 0, 255), -1)
                
                # Draw state info
                state_text = f"State: {state_output.state.name} | v_x: {v_x:.2f} | omega: {omega:.2f}"
                cv2.putText(display_frame, state_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Update global frame for web server
                with _frame_lock:
                    _latest_frame = display_frame

                # ===========================================================
                # Step 7: Visualization and logging
                # ===========================================================
                if self.visualizer is not None:
                    self.visualizer.update(
                        image=image,
                        detection=detection,
                        state=state_output.state,
                        v_x=v_x,
                        omega=omega
                    )

                    # Check for quit key
                    if self.visualizer.should_quit():
                        logger.info("Quit requested via visualization")
                        self._shutdown_requested = True

                # Print status (if verbose)
                if self.verbose and self._loop_count % 10 == 0:
                    self._print_status(
                        state_output.state,
                        detection,
                        v_x,
                        omega,
                        state_output.time_since_detection
                    )

                # Save frame (if enabled)
                if self.save_frames and detection is not None:
                    self._save_frame(image, detection)

                # ===========================================================
                # Step 7: Maintain loop rate
                # ===========================================================
                loop_duration = time.time() - loop_start
                sleep_time = self.control_period - loop_duration

                if sleep_time > 0:
                    time.sleep(sleep_time)
                elif loop_duration > self.control_period * 1.5:
                    logger.warning(
                        f"Control loop running slow: {loop_duration*1000:.1f}ms "
                        f"(target: {self.control_period*1000:.1f}ms)"
                    )

            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                # Stop robot on error
                try:
                    self.spot.stop()
                except:
                    pass
                time.sleep(0.5)

        logger.info("Exited control loop")

    def _print_status(
        self,
        state: BehaviorState,
        detection: Optional[Detection],
        v_x: float,
        omega: float,
        time_since_detection: float
    ) -> None:
        """
        Print current status to console.

        Args:
            state: Current behavior state
            detection: Current detection (or None)
            v_x: Current forward velocity command
            omega: Current angular velocity command
            time_since_detection: Time since last detection
        """
        det_str = "Yes" if detection else "No"
        if detection:
            det_str += f" (conf={detection.confidence:.2f}, area={detection.area:.0f})"

        print(
            f"[{self._loop_count:5d}] "
            f"State: {state.name:10s} | "
            f"Det: {det_str:30s} | "
            f"v_x: {v_x:+.2f} m/s | "
            f"omega: {omega:+.2f} rad/s | "
            f"t_lost: {time_since_detection:.1f}s"
        )

    def _save_frame(self, image: np.ndarray, detection: Detection) -> None:
        """
        Save a frame for debugging.

        Args:
            image: Camera image
            detection: Detection result
        """
        os.makedirs(self.save_path, exist_ok=True)
        filename = f"{self.save_path}/frame_{self._loop_count:06d}.jpg"
        cv2.imwrite(filename, image)

    def _shutdown(self) -> None:
        """
        Perform graceful shutdown sequence.

        1. Stop robot motion
        2. Sit down
        3. Release lease
        4. Clean up resources
        """
        logger.info("=" * 60)
        logger.info("Initiating graceful shutdown...")
        logger.info("=" * 60)

        self._running = False

        # Print final statistics
        if self._start_time is not None:
            runtime = time.time() - self._start_time
            avg_rate = self._loop_count / runtime if runtime > 0 else 0
            logger.info(f"Runtime: {runtime:.1f}s, Loops: {self._loop_count}, Avg rate: {avg_rate:.1f} Hz")

        # Stop robot
        try:
            logger.info("Stopping robot motion...")
            self.spot.stop()
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"Error stopping robot: {e}")

        # Sit down
        try:
            logger.info("Commanding robot to sit...")
            self.spot.sit()
            time.sleep(2.0)
        except Exception as e:
            logger.error(f"Error sitting robot: {e}")

        # Disconnect
        try:
            logger.info("Disconnecting from robot...")
            self.spot.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting: {e}")

        # Close ZED camera
        try:
            logger.info("Closing ZED camera...")
            self.zed_camera.close()
        except Exception as e:
            logger.error(f"Error closing ZED camera: {e}")

        # Close visualization
        if self.visualizer is not None:
            try:
                self.visualizer.close()
            except:
                pass

        logger.info("Shutdown complete")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Spot Person Follower - Visual Servoing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python src/main.py                           # Use default config
    python src/main.py --config custom.yaml      # Use custom config
    python src/main.py --verbose                 # Enable verbose output
        """
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create and run person follower
    follower = PersonFollower(config_path=args.config)
    follower.run()


if __name__ == "__main__":
    main()
