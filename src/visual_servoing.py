"""
Visual Servoing Module - Proportional Control Law
=================================================
This module implements the visual servoing control law that converts
person detection results into robot velocity commands.

Control Law Overview:
--------------------
Visual servoing uses feedback from the camera to control robot motion.
We implement a proportional (P) controller with two error signals:

1. LATERAL ERROR (controls turning):
   - Measures how far the person is from the image center horizontally
   - Error = (bbox_center_x - image_center_x) / image_width
   - Range: -0.5 (person on left) to +0.5 (person on right)
   - Positive error → person is to the right → turn right (negative omega)
   - Control: omega = -Kp_angular * lateral_error

2. DISTANCE ERROR (controls forward/backward motion):
   - Uses bounding box area as a proxy for distance
   - Larger bbox = person is closer, smaller bbox = person is farther
   - Error = (target_area - current_area) / target_area
   - Range: -∞ to +1 (clamped in practice)
   - Positive error → person is too far → move forward
   - Control: v_x = Kp_linear * distance_error

Coordinate Frame:
----------------
All velocity commands are in Spot's BODY frame:
    - v_x: Forward velocity (positive = forward)
    - v_y: Lateral velocity (positive = left, typically 0)
    - omega: Angular velocity (positive = counter-clockwise/left turn)

Safety Features:
---------------
- Velocity saturation (clipping to max limits)
- Deadband to prevent jitter from noise
- Velocity ramping for smooth acceleration

Author: Generated for Spot Person Follower Project
"""

import logging
from typing import Tuple, Optional
from dataclasses import dataclass

# Import Detection type from perception module
from .perception import Detection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ControlOutput:
    """
    Represents the output of the visual servoing controller.

    Attributes:
        v_x: Forward velocity command (m/s)
        v_y: Lateral velocity command (m/s), typically 0
        omega: Angular velocity command (rad/s)
        pitch: Body pitch command (rad) for tracking elevated targets
        lateral_error: Raw lateral error (for debugging)
        distance_error: Raw distance error (for debugging)
        pitch_error: Raw pitch error (for debugging)
    """
    v_x: float
    v_y: float
    omega: float
    pitch: float
    lateral_error: float
    distance_error: float
    pitch_error: float = 0.0


class VisualServoingController:
    """
    Proportional controller for person-following visual servoing.

    This controller computes velocity commands to keep a detected person
    centered in the camera view and at a target distance.

    Usage:
        controller = VisualServoingController(
            kp_linear=0.3,
            kp_angular=0.5,
            target_bbox_area=30000,
            image_width=640,
            image_height=480
        )
        output = controller.compute(detection)
        spot.move(output.v_x, output.v_y, output.omega)

    Control Behavior:
        - Person to the left → turn left (positive omega)
        - Person to the right → turn right (negative omega)
        - Person too far → move forward (positive v_x)
        - Person too close → move backward (negative v_x)
    """

    def __init__(
        self,
        kp_linear: float = 0.3,
        kp_angular: float = 0.5,
        target_bbox_area: float = 30000.0,
        image_width: int = 640,
        image_height: int = 480,
        max_linear_velocity: float = 0.5,
        max_angular_velocity: float = 0.6,
        deadband: float = 0.05,
        velocity_ramp_rate: float = 0.1,
        kp_pitch: float = 0.3,
        max_pitch_up: float = 0.26,
        max_pitch_down: float = 0.17
    ) -> None:
        """
        Initialize the visual servoing controller.

        Args:
            kp_linear: Proportional gain for forward/backward control.
                      Higher = more aggressive distance tracking.
                      Start with 0.1-0.3, increase if too slow.
            kp_angular: Proportional gain for turning control.
                       Higher = faster turning response.
                       Start with 0.3-0.5, increase if too slow.
            target_bbox_area: Desired bounding box area in pixels squared.
                             Determines following distance.
                             ~30000 for ~2m, ~50000 for ~1.5m.
            image_width: Camera image width in pixels.
            image_height: Camera image height in pixels.
            max_linear_velocity: Maximum forward/backward speed (m/s).
            max_angular_velocity: Maximum turning speed (rad/s).
            deadband: Error threshold below which no command is issued.
                     Prevents jitter from small detection variations.
            velocity_ramp_rate: Maximum velocity change per cycle (m/s).
                               Smooths acceleration/deceleration.
        """
        # Control gains
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular

        # Target configuration
        self.target_bbox_area = target_bbox_area

        # Image parameters
        self.image_width = image_width
        self.image_height = image_height
        self.image_center_x = image_width / 2.0
        self.image_center_y = image_height / 2.0

        # Safety limits
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity

        # Deadband and ramping
        self.deadband = deadband
        self.velocity_ramp_rate = velocity_ramp_rate

        # Pitch control parameters
        self.kp_pitch = kp_pitch
        self.max_pitch_up = max_pitch_up
        self.max_pitch_down = max_pitch_down

        # State for velocity ramping
        self._prev_v_x = 0.0
        self._prev_omega = 0.0
        self._prev_pitch = 0.0

        logger.info(
            f"VisualServoingController initialized: "
            f"Kp_lin={kp_linear}, Kp_ang={kp_angular}, "
            f"target_area={target_bbox_area}"
        )

    def compute(self, detection: Optional[Detection]) -> ControlOutput:
        """
        Compute velocity commands from a person detection.

        This is the main control law computation. It takes a detection
        and returns velocity commands to track the person.

        Args:
            detection: Person detection from the perception module.
                      If None, returns zero velocities.

        Returns:
            ControlOutput with velocity commands and error values.

        Control Law:
            1. Compute lateral error: how far person is from image center
            2. Compute distance error: how far from target bbox area
            3. Apply proportional control with gains
            4. Apply deadband to ignore small errors
            5. Saturate velocities to safety limits
            6. Apply velocity ramping for smooth motion
        """
        # No detection - stop moving
        if detection is None:
            return self._apply_ramping(ControlOutput(
                v_x=0.0,
                v_y=0.0,
                omega=0.0,
                pitch=0.0,
                lateral_error=0.0,
                distance_error=0.0,
                pitch_error=0.0
            ))

        # =================================================================
        # STEP 1: Compute Lateral Error
        # =================================================================
        # Lateral error measures horizontal offset from image center
        # Normalized to range [-0.5, 0.5]
        #
        # Error convention:
        #   Positive error = person is to the RIGHT of center
        #   Negative error = person is to the LEFT of center
        #
        # In image coordinates:
        #   x increases to the RIGHT
        #   So: error = (person_x - center_x) / width
        #
        lateral_error = (detection.center_x - self.image_center_x) / self.image_width

        # =================================================================
        # STEP 2: Compute Distance Error
        # =================================================================
        # Distance error uses bounding box area as proxy for distance
        # Normalized relative to target area
        #
        # Error convention:
        #   Positive error = person is TOO FAR (bbox smaller than target)
        #   Negative error = person is TOO CLOSE (bbox larger than target)
        #
        # Formula: error = (target_area - current_area) / target_area
        #
        # Example:
        #   target = 30000, current = 15000 → error = 0.5 (person too far)
        #   target = 30000, current = 45000 → error = -0.5 (person too close)
        #
        current_area = detection.area
        distance_error = (self.target_bbox_area - current_area) / self.target_bbox_area

        # Clamp distance error to reasonable range
        # This prevents extreme velocities when person is very close/far
        distance_error = max(-1.0, min(1.0, distance_error))

        # =================================================================
        # STEP 3: Apply Proportional Control
        # =================================================================
        # Linear velocity (forward/backward):
        #   v_x = Kp_linear * distance_error
        #   Positive error → positive v_x → move forward
        #
        v_x_raw = self.kp_linear * distance_error

        # Angular velocity (turning):
        #   omega = -Kp_angular * lateral_error
        #
        # The NEGATIVE sign is crucial! Here's why:
        #   - Positive lateral_error = person to the RIGHT
        #   - We want to turn RIGHT to face them
        #   - In Spot's body frame, turning RIGHT = NEGATIVE omega
        #   - So: omega = -Kp * (+error) = negative = turn right ✓
        #
        #   - Negative lateral_error = person to the LEFT
        #   - We want to turn LEFT to face them
        #   - Turning LEFT = POSITIVE omega
        #   - So: omega = -Kp * (-error) = positive = turn left ✓
        #
        omega_raw = -self.kp_angular * lateral_error

        # =================================================================
        # STEP 4: Apply Deadband
        # =================================================================
        # Deadband prevents jitter when errors are very small
        # Only command motion if error exceeds threshold
        #
        if abs(lateral_error) < self.deadband:
            omega_raw = 0.0

        if abs(distance_error) < self.deadband:
            v_x_raw = 0.0

        # =================================================================
        # STEP 5: Saturate Velocities (Safety Limits)
        # =================================================================
        # Clamp velocities to maximum allowed values
        # This is critical for safety!
        #
        v_x = self._saturate(v_x_raw, -self.max_linear_velocity, self.max_linear_velocity)
        omega = self._saturate(omega_raw, -self.max_angular_velocity, self.max_angular_velocity)

        # =================================================================
        # STEP 5.5: Compute Pitch Control
        # =================================================================
        # Pitch error measures vertical offset from image center
        # If person is HIGH in image (small y) → tilt UP (positive pitch)
        # If person is LOW in image (large y) → tilt DOWN (negative pitch)
        #
        # Error convention:
        #   Positive error = person is ABOVE center (needs to look UP)
        #   Negative error = person is BELOW center (needs to look DOWN)
        #
        pitch_error = (self.image_center_y - detection.center_y) / self.image_center_y
        
        # Apply proportional control to pitch
        pitch_raw = self.kp_pitch * pitch_error
        
        # Clamp pitch to safe range (asymmetric: can look up more than down)
        pitch = self._saturate(pitch_raw, -self.max_pitch_down, self.max_pitch_up)

        # Create output (before ramping, for logging)
        output = ControlOutput(
            v_x=v_x,
            v_y=0.0,  # No lateral movement for person following
            omega=omega,
            pitch=pitch,
            lateral_error=lateral_error,
            distance_error=distance_error,
            pitch_error=pitch_error
        )

        # =================================================================
        # STEP 6: Apply Velocity Ramping
        # =================================================================
        # Smooth acceleration/deceleration for comfort and stability
        #
        output = self._apply_ramping(output)

        logger.debug(
            f"Control: lat_err={lateral_error:.3f}, dist_err={distance_error:.3f}, "
            f"v_x={output.v_x:.2f}, omega={output.omega:.2f}"
        )

        return output

    def _saturate(self, value: float, min_val: float, max_val: float) -> float:
        """
        Clamp a value to a specified range.

        Args:
            value: Value to clamp
            min_val: Minimum allowed value
            max_val: Maximum allowed value

        Returns:
            Clamped value
        """
        return max(min_val, min(max_val, value))

    def _apply_ramping(self, output: ControlOutput) -> ControlOutput:
        """
        Apply velocity ramping for smooth acceleration.

        Limits how quickly velocities can change between control cycles.
        This prevents jerky motion and improves stability.

        Args:
            output: Raw control output

        Returns:
            Control output with ramped velocities
        """
        # Compute velocity change limits
        max_delta_v = self.velocity_ramp_rate
        max_delta_omega = self.velocity_ramp_rate * 2  # Allow faster angular changes

        # Ramp linear velocity
        delta_v_x = output.v_x - self._prev_v_x
        if abs(delta_v_x) > max_delta_v:
            delta_v_x = max_delta_v if delta_v_x > 0 else -max_delta_v
        v_x_ramped = self._prev_v_x + delta_v_x

        # Ramp angular velocity
        delta_omega = output.omega - self._prev_omega
        if abs(delta_omega) > max_delta_omega:
            delta_omega = max_delta_omega if delta_omega > 0 else -max_delta_omega
        omega_ramped = self._prev_omega + delta_omega

        # Ramp pitch
        max_delta_pitch = self.velocity_ramp_rate  # Same rate as linear velocity
        delta_pitch = output.pitch - self._prev_pitch
        if abs(delta_pitch) > max_delta_pitch:
            delta_pitch = max_delta_pitch if delta_pitch > 0 else -max_delta_pitch
        pitch_ramped = self._prev_pitch + delta_pitch

        # Update previous values for next cycle
        self._prev_v_x = v_x_ramped
        self._prev_omega = omega_ramped
        self._prev_pitch = pitch_ramped

        return ControlOutput(
            v_x=v_x_ramped,
            v_y=output.v_y,
            omega=omega_ramped,
            pitch=pitch_ramped,
            lateral_error=output.lateral_error,
            distance_error=output.distance_error,
            pitch_error=output.pitch_error
        )

    def reset(self) -> None:
        """
        Reset controller state.

        Call this when transitioning to a new tracking target or
        after losing detection for an extended period.
        """
        self._prev_v_x = 0.0
        self._prev_omega = 0.0
        self._prev_pitch = 0.0
        logger.info("Visual servoing controller reset")

    def set_gains(self, kp_linear: float, kp_angular: float) -> None:
        """
        Update control gains at runtime.

        Useful for tuning while the robot is running.

        Args:
            kp_linear: New linear gain
            kp_angular: New angular gain
        """
        self.kp_linear = kp_linear
        self.kp_angular = kp_angular
        logger.info(f"Gains updated: Kp_linear={kp_linear}, Kp_angular={kp_angular}")

    def set_target_area(self, target_area: float) -> None:
        """
        Update target bounding box area at runtime.

        Use this to adjust following distance while running.

        Args:
            target_area: New target area in pixels squared
        """
        self.target_bbox_area = target_area
        logger.info(f"Target bbox area updated: {target_area}")


def main():
    """
    Test the visual servoing controller with simulated detections.

    This demonstrates the control law behavior without requiring
    actual robot hardware or camera input.
    """
    print("Visual Servoing Controller Test")
    print("=" * 50)

    # Create controller with default settings
    controller = VisualServoingController(
        kp_linear=0.3,
        kp_angular=0.5,
        target_bbox_area=30000,
        image_width=640,
        image_height=480,
        max_linear_velocity=0.5,
        max_angular_velocity=0.6
    )

    # Test cases: (description, detection)
    test_cases = [
        ("Person centered, correct distance",
         Detection.from_bbox(270, 190, 370, 290, 0.9)),  # 100x100 = 10000 area, centered

        ("Person to the left",
         Detection.from_bbox(50, 190, 150, 290, 0.9)),   # Left side of image

        ("Person to the right",
         Detection.from_bbox(490, 190, 590, 290, 0.9)),  # Right side of image

        ("Person too far (small bbox)",
         Detection.from_bbox(300, 220, 340, 260, 0.9)),  # 40x40 = 1600 area

        ("Person too close (large bbox)",
         Detection.from_bbox(170, 90, 470, 390, 0.9)),   # 300x300 = 90000 area

        ("No detection",
         None),
    ]

    print("\nTest Results:")
    print("-" * 50)

    for description, detection in test_cases:
        output = controller.compute(detection)

        print(f"\n{description}:")
        if detection:
            print(f"  Detection: center=({detection.center_x:.0f}, {detection.center_y:.0f}), "
                  f"area={detection.area:.0f}")
        else:
            print("  Detection: None")

        print(f"  Lateral error: {output.lateral_error:+.3f}")
        print(f"  Distance error: {output.distance_error:+.3f}")
        print(f"  v_x (forward): {output.v_x:+.3f} m/s")
        print(f"  omega (turn): {output.omega:+.3f} rad/s")

        # Interpret the command
        if output.v_x > 0.01:
            print(f"  → Moving FORWARD")
        elif output.v_x < -0.01:
            print(f"  → Moving BACKWARD")

        if output.omega > 0.01:
            print(f"  → Turning LEFT")
        elif output.omega < -0.01:
            print(f"  → Turning RIGHT")

        # Reset for next test (to test ramping start from zero)
        controller.reset()


if __name__ == "__main__":
    main()
