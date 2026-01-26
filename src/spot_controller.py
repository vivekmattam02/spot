"""
Spot Controller Module
======================
This module provides a high-level interface to the Boston Dynamics Spot robot.
It handles all SDK interactions including authentication, lease management,
E-Stop, mobility commands, and image capture.

Coordinate Frame Reference:
--------------------------
Spot uses a body-centric coordinate frame:
    - X-axis: Points FORWARD (positive = robot moves forward)
    - Y-axis: Points LEFT (positive = robot moves left)
    - Z-axis: Points UP (positive = robot rises)

Angular velocity (omega/yaw):
    - Positive omega = counter-clockwise rotation (turn left)
    - Negative omega = clockwise rotation (turn right)

Author: Generated for Spot Person Follower Project
"""

import time
import logging
from typing import Optional, Tuple
import numpy as np
import cv2

# Boston Dynamics SDK imports
import bosdyn.client
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot import Robot
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2, estop_pb2
from bosdyn.client.exceptions import ResponseError, RpcError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpotController:
    """
    High-level controller for Boston Dynamics Spot robot.

    This class encapsulates all Spot SDK interactions and provides simple
    methods for common operations like standing, sitting, moving, and
    capturing images.

    Usage:
        controller = SpotController(hostname, username, password)
        controller.connect()
        controller.stand()
        controller.move(v_x=0.5, v_y=0.0, omega=0.0)
        image = controller.get_image()
        controller.sit()
        controller.disconnect()

    Attributes:
        hostname: IP address or hostname of the Spot robot
        username: Authentication username
        password: Authentication password
        camera_source: Name of camera to use for image capture
    """

    def __init__(
        self,
        hostname: str,
        username: str,
        password: str,
        camera_source: str = "frontleft_fisheye_image"
    ) -> None:
        """
        Initialize SpotController with connection parameters.

        Args:
            hostname: Robot's IP address (e.g., "192.168.80.3")
            username: Robot username (e.g., "admin")
            password: Robot password
            camera_source: Camera to capture images from
        """
        self.hostname = hostname
        self.username = username
        self.password = password
        self.camera_source = camera_source

        # SDK objects - initialized during connect()
        self._sdk: Optional[bosdyn.client.Sdk] = None
        self._robot: Optional[Robot] = None
        self._robot_state_client: Optional[RobotStateClient] = None
        self._robot_command_client: Optional[RobotCommandClient] = None
        self._lease_client: Optional[LeaseClient] = None
        self._lease: Optional[object] = None
        self._lease_keepalive: Optional[LeaseKeepAlive] = None
        self._estop_client: Optional[EstopClient] = None
        self._estop_endpoint: Optional[EstopEndpoint] = None
        self._estop_keepalive: Optional[EstopKeepAlive] = None
        self._image_client: Optional[ImageClient] = None

        # State tracking
        self._is_connected: bool = False
        self._is_powered_on: bool = False
        self._is_standing: bool = False

        # Velocity ramping state (for smooth acceleration)
        self._current_v_x: float = 0.0
        self._current_v_y: float = 0.0
        self._current_omega: float = 0.0

        logger.info(f"SpotController initialized for robot at {hostname}")

    def connect(self) -> bool:
        """
        Connect to Spot robot and acquire necessary clients and lease.

        This method performs the complete connection sequence:
        1. Create SDK and robot objects
        2. Authenticate with the robot
        3. Establish time sync
        4. Create all required clients
        5. Acquire lease (control authority)
        6. Set up E-Stop endpoint

        Returns:
            True if connection successful, False otherwise

        Raises:
            RuntimeError: If connection fails at any step
        """
        try:
            logger.info(f"Connecting to Spot at {self.hostname}...")

            # Step 1: Create SDK instance
            # The SDK is the main entry point for all Spot interactions
            self._sdk = create_standard_sdk("SpotPersonFollower")

            # Step 2: Create robot object and authenticate
            # This establishes the connection to the physical robot
            self._robot = self._sdk.create_robot(self.hostname)

            logger.info("Authenticating with robot...")
            # Authenticate using provided credentials
            self._robot.authenticate(self.username, self.password)

            # Step 3: Establish time synchronization
            # Critical for command timing - robot rejects commands with bad timestamps
            logger.info("Synchronizing time with robot...")
            self._robot.time_sync.wait_for_sync()

            # Step 4: Create service clients
            # Each client handles a specific robot subsystem
            logger.info("Creating service clients...")

            # Robot state client - for reading robot status
            self._robot_state_client = self._robot.ensure_client(
                RobotStateClient.default_service_name
            )

            # Robot command client - for sending movement commands
            self._robot_command_client = self._robot.ensure_client(
                RobotCommandClient.default_service_name
            )

            # Lease client - for acquiring control authority
            self._lease_client = self._robot.ensure_client(
                LeaseClient.default_service_name
            )

            # Image client - for capturing camera images
            self._image_client = self._robot.ensure_client(
                ImageClient.default_service_name
            )

            # E-Stop client - for emergency stop functionality
            self._estop_client = self._robot.ensure_client(
                EstopClient.default_service_name
            )

            # Step 5: Acquire lease
            # The lease gives us exclusive control of the robot
            # Only one client can hold the lease at a time
            logger.info("Acquiring lease...")
            self._lease = self._lease_client.take()

            # Start lease keepalive - automatically renews lease
            self._lease_keepalive = LeaseKeepAlive(
                self._lease_client,
                must_acquire=True,
                return_at_exit=True
            )

            # Step 6: Set up E-Stop
            # Software E-Stop allows us to stop the robot programmatically
            logger.info("Setting up E-Stop endpoint...")
            self._setup_estop()

            self._is_connected = True
            logger.info("Successfully connected to Spot!")

            return True

        except ResponseError as e:
            logger.error(f"Robot response error during connection: {e}")
            raise RuntimeError(f"Failed to connect to Spot: {e}")
        except RpcError as e:
            logger.error(f"RPC error during connection: {e}")
            raise RuntimeError(f"Network error connecting to Spot: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during connection: {e}")
            raise RuntimeError(f"Failed to connect to Spot: {e}")

    def _setup_estop(self) -> None:
        """
        Set up software E-Stop endpoint.

        The E-Stop system is a critical safety feature. This creates a
        software E-Stop endpoint that must be kept alive for the robot
        to accept movement commands.

        If the E-Stop keepalive stops (e.g., program crashes), the robot
        will automatically stop moving - this is a safety feature.
        """
        # Check if there's an existing E-Stop configuration
        estop_config = self._estop_client.get_config()

        # Create our E-Stop endpoint
        # This registers us as an E-Stop authority
        self._estop_endpoint = EstopEndpoint(
            client=self._estop_client,
            name="person_follower_estop",
            estop_timeout=9.0  # Robot stops if no keepalive for 9 seconds
        )

        # Force register our endpoint
        # This may replace existing endpoints - be careful in production
        self._estop_endpoint.force_simple_setup()

        # Start E-Stop keepalive thread
        # This sends periodic check-ins to keep E-Stop active
        self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)

        logger.info("E-Stop endpoint configured and active")

    def check_estop(self) -> bool:
        """
        Check if E-Stop is engaged (robot stopped).

        Returns:
            True if E-Stop is NOT engaged (robot can move)
            False if E-Stop IS engaged (robot stopped)
        """
        if self._estop_client is None:
            return False

        try:
            estop_status = self._estop_client.get_status()
            # ESTOP_LEVEL_NONE means no E-Stop is active
            # Use the protobuf enum for comparison
            return estop_status.stop_level == estop_pb2.ESTOP_LEVEL_NONE
        except Exception as e:
            logger.error(f"Error checking E-Stop status: {e}")
            return False

    def trigger_estop(self) -> None:
        """
        Trigger the software E-Stop to immediately stop the robot.

        Use this in emergency situations to halt all robot motion.
        The robot will need to be manually reset after E-Stop.
        """
        if self._estop_endpoint is not None:
            logger.warning("TRIGGERING E-STOP!")
            self._estop_endpoint.stop()

    def release_estop(self) -> None:
        """
        Release the software E-Stop to allow robot motion.

        Call this after an E-Stop to resume normal operation.
        Note: Hardware E-Stops must be released manually.
        """
        if self._estop_endpoint is not None:
            logger.info("Releasing E-Stop")
            self._estop_endpoint.allow()

    def power_on(self) -> bool:
        """
        Power on the robot's motors.

        The robot must be powered on before it can stand or move.
        This can take several seconds as the robot initializes.

        Returns:
            True if power on successful
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to robot. Call connect() first.")

        try:
            logger.info("Powering on robot motors...")
            self._robot.power_on(timeout_sec=20)

            # Verify power is on
            assert self._robot.is_powered_on(), "Failed to power on"

            self._is_powered_on = True
            logger.info("Robot motors powered on")
            return True

        except Exception as e:
            logger.error(f"Failed to power on robot: {e}")
            raise

    def power_off(self) -> bool:
        """
        Safely power off the robot's motors.

        The robot will sit down before powering off if it's standing.

        Returns:
            True if power off successful
        """
        if not self._is_connected:
            return True

        try:
            logger.info("Powering off robot motors...")
            self._robot.power_off(cut_immediately=False, timeout_sec=20)
            self._is_powered_on = False
            self._is_standing = False
            logger.info("Robot motors powered off")
            return True
        except Exception as e:
            logger.error(f"Error powering off: {e}")
            return False

    def stand(self) -> bool:
        """
        Command the robot to stand up.

        The robot must be powered on before standing.
        This is a blocking call that waits for the stand to complete.

        Returns:
            True if stand successful
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to robot")

        if not self._is_powered_on:
            logger.info("Robot not powered on, powering on first...")
            self.power_on()

        try:
            logger.info("Commanding robot to stand...")

            # Create stand command
            # blocking_stand waits until the robot is fully standing
            blocking_stand(
                self._robot_command_client,
                timeout_sec=10
            )

            self._is_standing = True
            logger.info("Robot is now standing")
            return True

        except Exception as e:
            logger.error(f"Failed to stand: {e}")
            raise

    def sit(self) -> bool:
        """
        Command the robot to sit down safely.

        This is a blocking call that waits for the sit to complete.
        Always call this before disconnecting or powering off.

        Returns:
            True if sit successful
        """
        if not self._is_connected or not self._is_standing:
            return True

        try:
            logger.info("Commanding robot to sit...")

            # Create sit (safe power off) command
            sit_command = RobotCommandBuilder.safe_power_off_command()

            # Issue the command
            self._robot_command_client.robot_command(
                lease=None,  # Uses current lease
                command=sit_command,
                end_time_secs=time.time() + 10
            )

            # Wait for sit to complete
            time.sleep(3.0)

            self._is_standing = False
            logger.info("Robot is now sitting")
            return True

        except Exception as e:
            logger.error(f"Failed to sit: {e}")
            return False

    def move(
        self,
        v_x: float,
        v_y: float,
        omega: float,
        pitch: float = 0.0,
        duration: float = 0.5
    ) -> bool:
        """
        Command the robot to move with specified velocities and body pitch.

        This sends a velocity command in the robot's body frame.
        Commands are executed for the specified duration, after which
        the robot will stop if no new command is received.

        Coordinate Frame (Body Frame):
            v_x: Forward velocity (positive = forward, negative = backward)
            v_y: Lateral velocity (positive = left, negative = right)
            omega: Angular velocity (positive = turn left, negative = turn right)
            pitch: Body pitch angle in radians (positive = tilt up, negative = tilt down)

        Args:
            v_x: Forward velocity in m/s
            v_y: Lateral velocity in m/s (usually 0 for person following)
            omega: Angular velocity in rad/s
            pitch: Body pitch in radians (for tracking elevated/lowered targets)
            duration: How long this command should execute (seconds)

        Returns:
            True if command sent successfully

        Note:
            For smooth motion, call this method repeatedly at ~10Hz
            with updated velocities.
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to robot")

        if not self._is_standing:
            logger.warning("Robot not standing, cannot move")
            return False

        # Safety check: Verify E-Stop is not engaged
        if not self.check_estop():
            logger.warning("E-Stop is engaged, cannot move")
            return False

        try:
            # Import helpers for body orientation
            from bosdyn.client.math_helpers import Quat
            from bosdyn.api import geometry_pb2, trajectory_pb2
            from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
            
            # Create quaternion for pitch rotation
            # Quat.from_pitch creates a quaternion for rotation about the Y axis
            pitch_quat = Quat.from_pitch(pitch)
            
            # Build body control params with pitch
            # This sets the body orientation relative to the footprint frame
            position = geometry_pb2.Vec3(z=0.0)  # No height offset
            rotation = geometry_pb2.Quaternion(
                w=pitch_quat.w, x=pitch_quat.x, y=pitch_quat.y, z=pitch_quat.z
            )
            pose = geometry_pb2.SE3Pose(position=position, rotation=rotation)
            point = trajectory_pb2.SE3TrajectoryPoint(pose=pose)
            traj = trajectory_pb2.SE3Trajectory(points=[point])
            body_control = spot_command_pb2.BodyControlParams(base_offset_rt_footprint=traj)
            
            # Create mobility params with body control
            params = spot_command_pb2.MobilityParams(body_control=body_control)
            
            # Build velocity command with body orientation params
            # synchro_velocity_command creates a synchronized locomotion command
            # that moves all legs together smoothly
            velocity_command = RobotCommandBuilder.synchro_velocity_command(
                v_x=v_x,      # Forward velocity (m/s)
                v_y=v_y,      # Lateral velocity (m/s)
                v_rot=omega,  # Angular velocity (rad/s)
                params=params # Body orientation params
            )

            # Calculate end time for this command
            # The robot will execute this command until end_time
            end_time = time.time() + duration

            # Send the command to the robot
            self._robot_command_client.robot_command(
                lease=None,  # Uses current lease automatically
                command=velocity_command,
                end_time_secs=end_time
            )

            # Update current velocity state (for ramping)
            self._current_v_x = v_x
            self._current_v_y = v_y
            self._current_omega = omega

            return True

        except ResponseError as e:
            logger.error(f"Command rejected by robot: {e}")
            return False
        except RpcError as e:
            logger.error(f"Network error sending command: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending move command: {e}")
            return False

    def stop(self) -> bool:
        """
        Stop all robot motion immediately.

        Sends a zero-velocity command to halt the robot.

        Returns:
            True if stop command sent successfully
        """
        logger.info("Stopping robot motion")
        return self.move(0.0, 0.0, 0.0)

    def get_image(self) -> Optional[np.ndarray]:
        """
        Capture an image from the robot's camera.

        Gets a single frame from the configured camera source.
        The image is decoded and returned as a numpy array.

        Returns:
            numpy.ndarray: BGR image (OpenCV format) or None if capture fails

        Note:
            Spot's fisheye cameras have significant distortion at the edges.
            For person detection, the center region is most reliable.
        """
        if not self._is_connected:
            raise RuntimeError("Not connected to robot")

        try:
            # Build image request for the specified camera source
            # Using the image client's get_image_from_sources method
            image_responses = self._image_client.get_image_from_sources(
                [self.camera_source]
            )

            if not image_responses:
                logger.warning("No image received from robot")
                return None

            image_response = image_responses[0]

            # Check the pixel format and decode accordingly
            image = image_response.shot.image

            if image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                # RGB format - convert to BGR for OpenCV
                img_array = np.frombuffer(image.data, dtype=np.uint8)
                img_array = img_array.reshape((image.rows, image.cols, 3))
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return img_bgr

            elif image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                # Grayscale - convert to BGR
                img_array = np.frombuffer(image.data, dtype=np.uint8)
                img_array = img_array.reshape((image.rows, image.cols))
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                return img_bgr

            elif image.format == image_pb2.Image.FORMAT_JPEG:
                # JPEG compressed - decode
                np_arr = np.frombuffer(image.data, dtype=np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                return img_bgr

            else:
                # Try to decode as JPEG (common format)
                np_arr = np.frombuffer(image.data, dtype=np.uint8)
                img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if img_bgr is None:
                    logger.warning(f"Unknown image format: {image.pixel_format}")
                    return None

                return img_bgr

        except Exception as e:
            logger.error(f"Error capturing image: {e}")
            return None

    def get_robot_state(self) -> dict:
        """
        Get current robot state information.

        Returns:
            Dictionary containing robot state information including
            battery level, motor states, and fault status.
        """
        if not self._is_connected:
            return {}

        try:
            state = self._robot_state_client.get_robot_state()

            return {
                "battery_percentage": state.power_state.locomotion_charge_percentage.value,
                "is_powered_on": self._is_powered_on,
                "is_standing": self._is_standing,
                "estop_engaged": not self.check_estop(),
            }
        except Exception as e:
            logger.error(f"Error getting robot state: {e}")
            return {}

    def disconnect(self) -> None:
        """
        Safely disconnect from the robot.

        This method performs a clean shutdown sequence:
        1. Stop any motion
        2. Sit down (if standing)
        3. Release E-Stop keepalive
        4. Release lease

        Always call this method when done with the robot.
        """
        logger.info("Disconnecting from Spot...")

        try:
            # Stop motion
            if self._is_standing:
                self.stop()
                time.sleep(0.5)
                self.sit()

            # Power off
            if self._is_powered_on:
                self.power_off()

        except Exception as e:
            logger.warning(f"Error during shutdown sequence: {e}")

        # Release E-Stop keepalive
        if self._estop_keepalive is not None:
            try:
                self._estop_keepalive.shutdown()
                self._estop_keepalive = None
            except Exception as e:
                logger.warning(f"Error releasing E-Stop keepalive: {e}")

        # Release lease keepalive
        if self._lease_keepalive is not None:
            try:
                self._lease_keepalive.shutdown()
                self._lease_keepalive = None
            except Exception as e:
                logger.warning(f"Error releasing lease keepalive: {e}")

        self._is_connected = False
        self._is_powered_on = False
        self._is_standing = False

        logger.info("Disconnected from Spot")

    def __enter__(self) -> "SpotController":
        """Context manager entry - connect to robot."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - disconnect from robot."""
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        """Check if currently connected to the robot."""
        return self._is_connected

    @property
    def is_standing(self) -> bool:
        """Check if robot is currently standing."""
        return self._is_standing


def main():
    """
    Simple test of SpotController functionality.

    This demonstrates basic connection and status checking.
    Does not actually move the robot - use test_mobility.py for that.
    """
    print("SpotController Test")
    print("=" * 40)

    # For testing, you can use these placeholder values
    # In production, load from config
    hostname = "192.168.80.3"
    username = "admin"
    password = "password"

    controller = SpotController(hostname, username, password)

    try:
        print(f"Connecting to Spot at {hostname}...")
        controller.connect()
        print("Connected!")

        state = controller.get_robot_state()
        print(f"Battery: {state.get('battery_percentage', 'N/A')}%")
        print(f"E-Stop engaged: {state.get('estop_engaged', 'N/A')}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        controller.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
