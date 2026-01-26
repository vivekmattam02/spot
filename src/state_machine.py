"""
State Machine Module - Behavior Management
==========================================
This module implements the behavior state machine that controls
the high-level behavior of the person-following system.

State Overview:
--------------
The system operates in two primary states:

1. TRACKING State:
   - Active when a person is detected with sufficient confidence
   - Visual servoing controller generates velocity commands
   - Robot actively follows the detected person

2. SEARCH State:
   - Activated when no person detected for >detection_timeout seconds
   - Robot rotates slowly to scan for people
   - Returns to TRACKING when person is found

State Transitions:
-----------------
    ┌─────────────┐        Detection found        ┌─────────────┐
    │   SEARCH    │ ────────────────────────────> │  TRACKING   │
    │ (rotate)    │                               │ (follow)    │
    └─────────────┘ <──────────────────────────── └─────────────┘
                      No detection for timeout

Safety Behavior:
---------------
- Lost detection while tracking → immediately stop, then transition to SEARCH
- Max search duration → stop rotating (prevents infinite spinning)
- Emergency stop → handled at controller level

Author: Generated for Spot Person Follower Project
"""

import time
import logging
from enum import Enum, auto
from typing import Optional, Tuple
from dataclasses import dataclass

# Import Detection type
from .perception import Detection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehaviorState(Enum):
    """
    Enumeration of behavior states for the person follower.

    States:
        TRACKING: Actively following a detected person
        SEARCH: Rotating to search for a person
        STOPPED: Not moving (initial state or after timeout)
    """
    TRACKING = auto()
    SEARCH = auto()
    STOPPED = auto()


@dataclass
class StateOutput:
    """
    Output from the state machine update.

    Attributes:
        state: Current behavior state
        should_servo: If True, use visual servoing controller
        search_omega: If in SEARCH state, the angular velocity to use
        time_since_detection: Seconds since last valid detection
        search_duration: Seconds spent in current SEARCH state
    """
    state: BehaviorState
    should_servo: bool
    search_omega: float
    time_since_detection: float
    search_duration: float


class BehaviorStateMachine:
    """
    State machine for managing person-following behavior.

    This class tracks detections over time and determines whether
    the robot should be tracking (following a person) or searching
    (rotating to find a person).

    Usage:
        state_machine = BehaviorStateMachine(
            detection_timeout=2.0,
            search_angular_velocity=0.2
        )

        # In control loop:
        output = state_machine.update(detection)
        if output.state == BehaviorState.TRACKING:
            # Use visual servoing commands
            v_x, omega = servoing_controller.compute(detection)
        elif output.state == BehaviorState.SEARCH:
            # Rotate to search
            v_x, omega = 0.0, output.search_omega

    Attributes:
        detection_timeout: Seconds without detection before SEARCH
        search_angular_velocity: Rotation speed in SEARCH state (rad/s)
        max_search_duration: Maximum time to search before stopping
    """

    def __init__(
        self,
        detection_timeout: float = 2.0,
        search_angular_velocity: float = 0.2,
        max_search_duration: float = 30.0
    ) -> None:
        """
        Initialize the behavior state machine.

        Args:
            detection_timeout: Time in seconds without a detection
                              before transitioning to SEARCH state.
                              Default 2.0s prevents brief occlusions
                              from triggering search.
            search_angular_velocity: Angular velocity (rad/s) for
                                    rotation in SEARCH state.
                                    Positive = counter-clockwise.
                                    Default 0.2 rad/s ≈ 11 deg/s.
            max_search_duration: Maximum time to stay in SEARCH state
                                before stopping. Prevents endless spinning.
                                Default 30.0s = ~1.7 full rotations.
        """
        self.detection_timeout = detection_timeout
        self.search_angular_velocity = search_angular_velocity
        self.max_search_duration = max_search_duration

        # State tracking
        self._current_state = BehaviorState.STOPPED
        self._last_detection_time: Optional[float] = None
        self._search_start_time: Optional[float] = None
        self._last_detection: Optional[Detection] = None

        # Statistics
        self._tracking_time = 0.0
        self._search_time = 0.0
        self._state_transitions = 0

        logger.info(
            f"BehaviorStateMachine initialized: "
            f"timeout={detection_timeout}s, "
            f"search_omega={search_angular_velocity} rad/s"
        )

    def update(self, detection: Optional[Detection]) -> StateOutput:
        """
        Update the state machine with a new detection result.

        This method should be called every control cycle with the
        latest detection (or None if no detection).

        Args:
            detection: Person detection from perception module,
                      or None if no person detected.

        Returns:
            StateOutput containing current state and guidance for
            what velocity commands to use.

        State Transition Logic:
            1. If detection is valid:
               - Update last detection time
               - Transition to TRACKING if not already
            2. If no detection:
               - Check time since last detection
               - If > timeout: transition to SEARCH
            3. If in SEARCH and max duration exceeded:
               - Transition to STOPPED
        """
        current_time = time.time()

        # Track detection status
        has_detection = detection is not None and detection.confidence > 0.0

        # Calculate time since last detection
        time_since_detection = 0.0
        if self._last_detection_time is not None:
            time_since_detection = current_time - self._last_detection_time

        # Calculate search duration (if in SEARCH state)
        search_duration = 0.0
        if self._search_start_time is not None:
            search_duration = current_time - self._search_start_time

        # =========================================================
        # State Transition Logic
        # =========================================================

        if has_detection:
            # Valid detection received
            self._last_detection_time = current_time
            self._last_detection = detection

            if self._current_state != BehaviorState.TRACKING:
                self._transition_to(BehaviorState.TRACKING)
                self._search_start_time = None  # Reset search timer

            return StateOutput(
                state=BehaviorState.TRACKING,
                should_servo=True,
                search_omega=0.0,
                time_since_detection=0.0,
                search_duration=0.0
            )

        else:
            # No detection
            if self._current_state == BehaviorState.TRACKING:
                # Was tracking, check if we should transition to SEARCH
                if time_since_detection > self.detection_timeout:
                    self._transition_to(BehaviorState.SEARCH)
                    self._search_start_time = current_time
                    search_duration = 0.0
                else:
                    # Still within timeout, stay in TRACKING but stop moving
                    # This prevents the robot from blindly continuing
                    return StateOutput(
                        state=BehaviorState.TRACKING,
                        should_servo=False,  # Don't servo without detection
                        search_omega=0.0,
                        time_since_detection=time_since_detection,
                        search_duration=0.0
                    )

            if self._current_state == BehaviorState.SEARCH:
                # In SEARCH state
                if search_duration > self.max_search_duration:
                    # Exceeded max search time, stop
                    self._transition_to(BehaviorState.STOPPED)
                    return StateOutput(
                        state=BehaviorState.STOPPED,
                        should_servo=False,
                        search_omega=0.0,
                        time_since_detection=time_since_detection,
                        search_duration=search_duration
                    )
                else:
                    # Continue searching
                    return StateOutput(
                        state=BehaviorState.SEARCH,
                        should_servo=False,
                        search_omega=self.search_angular_velocity,
                        time_since_detection=time_since_detection,
                        search_duration=search_duration
                    )

            if self._current_state == BehaviorState.STOPPED:
                # In STOPPED state, check if we should start searching
                # This allows recovery from STOPPED state
                if self._last_detection_time is None:
                    # Never had a detection, start searching
                    self._transition_to(BehaviorState.SEARCH)
                    self._search_start_time = current_time
                    return StateOutput(
                        state=BehaviorState.SEARCH,
                        should_servo=False,
                        search_omega=self.search_angular_velocity,
                        time_since_detection=time_since_detection,
                        search_duration=0.0
                    )

            # Default: stay stopped
            return StateOutput(
                state=self._current_state,
                should_servo=False,
                search_omega=0.0,
                time_since_detection=time_since_detection,
                search_duration=search_duration
            )

    def _transition_to(self, new_state: BehaviorState) -> None:
        """
        Transition to a new state with logging.

        Args:
            new_state: The state to transition to
        """
        old_state = self._current_state
        self._current_state = new_state
        self._state_transitions += 1

        logger.info(
            f"State transition: {old_state.name} → {new_state.name} "
            f"(transition #{self._state_transitions})"
        )

    def get_state(self) -> BehaviorState:
        """
        Get the current behavior state.

        Returns:
            Current BehaviorState
        """
        return self._current_state

    def get_last_detection(self) -> Optional[Detection]:
        """
        Get the last valid detection.

        Returns:
            Last Detection object, or None if never detected
        """
        return self._last_detection

    def reset(self) -> None:
        """
        Reset the state machine to initial conditions.

        Call this when restarting the person-following behavior
        or after an emergency stop.
        """
        self._current_state = BehaviorState.STOPPED
        self._last_detection_time = None
        self._search_start_time = None
        self._last_detection = None

        logger.info("State machine reset to STOPPED state")

    def force_state(self, state: BehaviorState) -> None:
        """
        Force transition to a specific state.

        Use with caution - mainly for testing or emergency recovery.

        Args:
            state: State to force transition to
        """
        logger.warning(f"Forcing state transition to {state.name}")
        self._transition_to(state)

        if state == BehaviorState.SEARCH:
            self._search_start_time = time.time()

    def get_stats(self) -> dict:
        """
        Get state machine statistics.

        Returns:
            Dictionary with state statistics
        """
        return {
            "current_state": self._current_state.name,
            "state_transitions": self._state_transitions,
            "has_detection_history": self._last_detection_time is not None,
        }


def main():
    """
    Test the behavior state machine with simulated detections.

    This demonstrates state transitions without requiring
    actual robot hardware.
    """
    print("Behavior State Machine Test")
    print("=" * 50)

    # Create state machine
    sm = BehaviorStateMachine(
        detection_timeout=2.0,
        search_angular_velocity=0.2,
        max_search_duration=10.0  # Short for testing
    )

    print("\nSimulating detection sequence...")
    print("-" * 50)

    # Test sequence: simulate detection on/off pattern
    test_sequence = [
        # (time_offset, has_detection, description)
        (0.0, True, "Initial detection"),
        (0.5, True, "Continued tracking"),
        (1.0, True, "Continued tracking"),
        (1.5, False, "Lost detection"),
        (2.0, False, "Still lost (within timeout)"),
        (2.5, False, "Still lost (within timeout)"),
        (3.5, False, "Lost >2s - should SEARCH"),
        (4.0, False, "Searching..."),
        (5.0, True, "Found again!"),
        (5.5, True, "Back to tracking"),
    ]

    # Simulated detection
    mock_detection = Detection.from_bbox(270, 190, 370, 290, 0.9)

    start_time = time.time()

    for time_offset, has_detection, description in test_sequence:
        # Wait until simulated time
        target_time = start_time + time_offset
        if time.time() < target_time:
            time.sleep(target_time - time.time())

        # Update state machine
        detection = mock_detection if has_detection else None
        output = sm.update(detection)

        # Print result
        print(f"\nt={time_offset:.1f}s: {description}")
        print(f"  Detection: {'Yes' if has_detection else 'No'}")
        print(f"  State: {output.state.name}")
        print(f"  Should servo: {output.should_servo}")
        print(f"  Time since detection: {output.time_since_detection:.1f}s")

        if output.state == BehaviorState.SEARCH:
            print(f"  Search omega: {output.search_omega:.2f} rad/s")
            print(f"  Search duration: {output.search_duration:.1f}s")

    print("\n" + "=" * 50)
    print("Final Stats:", sm.get_stats())


if __name__ == "__main__":
    main()
