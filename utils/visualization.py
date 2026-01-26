"""
Visualization Module - Debug Display
====================================
This module provides real-time visualization of the person-following
system for debugging and monitoring.

Features:
---------
- Live camera feed display
- Bounding box overlay for detections
- State indicator (TRACKING/SEARCH/STOPPED)
- Velocity command display
- Center crosshair for reference
- FPS counter

Usage:
------
    visualizer = Visualizer()
    visualizer.update(image, detection, state, v_x, omega)
    if visualizer.should_quit():
        break
    visualizer.close()

Author: Generated for Spot Person Follower Project
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import time

# Import types from parent module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from perception import Detection
    from state_machine import BehaviorState
except ImportError:
    # Define minimal types if imports fail
    Detection = object
    BehaviorState = object


class Visualizer:
    """
    Real-time visualization for the person-following system.

    Displays the camera feed with overlaid information including
    detections, state, and velocity commands.

    The display includes:
    - Camera image with detection bounding box
    - Center crosshair (target for centering person)
    - Current state (TRACKING/SEARCH/STOPPED) with color coding
    - Velocity commands (v_x, omega) with visual bars
    - Detection confidence and area
    - FPS counter
    """

    # Color definitions (BGR format for OpenCV)
    COLOR_GREEN = (0, 255, 0)
    COLOR_RED = (0, 0, 255)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_BLUE = (255, 0, 0)
    COLOR_WHITE = (255, 255, 255)
    COLOR_BLACK = (0, 0, 0)
    COLOR_ORANGE = (0, 165, 255)

    # State colors
    STATE_COLORS = {
        "TRACKING": (0, 255, 0),     # Green
        "SEARCH": (0, 255, 255),     # Yellow
        "STOPPED": (0, 0, 255),      # Red
    }

    def __init__(
        self,
        window_name: str = "Spot Person Follower",
        image_width: int = 640,
        image_height: int = 480
    ) -> None:
        """
        Initialize the visualizer.

        Args:
            window_name: Name for the OpenCV window
            image_width: Expected image width
            image_height: Expected image height
        """
        self.window_name = window_name
        self.image_width = image_width
        self.image_height = image_height

        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, image_width, image_height)

        # FPS tracking
        self._frame_count = 0
        self._fps_start_time = time.time()
        self._current_fps = 0.0

        # Last key pressed
        self._last_key = -1

    def update(
        self,
        image: np.ndarray,
        detection: Optional[Detection] = None,
        state: Optional[BehaviorState] = None,
        v_x: float = 0.0,
        omega: float = 0.0,
        additional_info: Optional[dict] = None
    ) -> None:
        """
        Update the visualization with new data.

        Args:
            image: Camera image (BGR format)
            detection: Person detection (or None)
            state: Current behavior state
            v_x: Current forward velocity command (m/s)
            omega: Current angular velocity command (rad/s)
            additional_info: Optional dict with extra info to display
        """
        if image is None:
            return

        # Make a copy to avoid modifying original
        display = image.copy()

        # Update FPS
        self._update_fps()

        # Draw center crosshair
        self._draw_crosshair(display)

        # Draw detection bounding box
        if detection is not None:
            self._draw_detection(display, detection)

        # Draw state indicator
        state_name = state.name if hasattr(state, 'name') else str(state)
        self._draw_state(display, state_name)

        # Draw velocity bars
        self._draw_velocity_bars(display, v_x, omega)

        # Draw FPS
        self._draw_fps(display)

        # Draw additional info
        if additional_info:
            self._draw_info(display, additional_info)

        # Show the frame
        cv2.imshow(self.window_name, display)

        # Process key events (non-blocking)
        self._last_key = cv2.waitKey(1) & 0xFF

    def _update_fps(self) -> None:
        """Update FPS calculation."""
        self._frame_count += 1
        elapsed = time.time() - self._fps_start_time

        if elapsed >= 1.0:
            self._current_fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_start_time = time.time()

    def _draw_crosshair(self, image: np.ndarray) -> None:
        """
        Draw center crosshair on image.

        This shows the target location for centering the person.
        """
        h, w = image.shape[:2]
        cx, cy = w // 2, h // 2

        # Draw crosshair
        size = 30
        thickness = 1
        color = self.COLOR_WHITE

        # Horizontal line
        cv2.line(image, (cx - size, cy), (cx + size, cy), color, thickness)
        # Vertical line
        cv2.line(image, (cx, cy - size), (cx, cy + size), color, thickness)
        # Center dot
        cv2.circle(image, (cx, cy), 3, color, -1)

    def _draw_detection(self, image: np.ndarray, detection: Detection) -> None:
        """
        Draw bounding box and info for detection.

        Args:
            image: Image to draw on
            detection: Detection to visualize
        """
        # Get bounding box coordinates
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        cx, cy = int(detection.center_x), int(detection.center_y)

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), self.COLOR_GREEN, 2)

        # Draw center point
        cv2.circle(image, (cx, cy), 5, self.COLOR_RED, -1)

        # Draw line from image center to detection center
        h, w = image.shape[:2]
        img_cx, img_cy = w // 2, h // 2
        cv2.line(image, (img_cx, img_cy), (cx, cy), self.COLOR_YELLOW, 1)

        # Draw detection info
        info_text = f"Conf: {detection.confidence:.2f} Area: {detection.area:.0f}"
        cv2.putText(
            image, info_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            self.COLOR_GREEN, 1
        )

    def _draw_state(self, image: np.ndarray, state_name: str) -> None:
        """
        Draw state indicator in top-left corner.

        Args:
            image: Image to draw on
            state_name: Name of current state
        """
        # Get color for state
        color = self.STATE_COLORS.get(state_name, self.COLOR_WHITE)

        # Draw state background
        text = f"State: {state_name}"
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )

        cv2.rectangle(
            image,
            (10, 10),
            (20 + text_w, 20 + text_h + 10),
            self.COLOR_BLACK,
            -1
        )

        # Draw state text
        cv2.putText(
            image, text,
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            color, 2
        )

    def _draw_velocity_bars(
        self,
        image: np.ndarray,
        v_x: float,
        omega: float,
        max_v: float = 0.5,
        max_omega: float = 0.6
    ) -> None:
        """
        Draw velocity command bars at bottom of image.

        Shows visual representation of commanded velocities.

        Args:
            image: Image to draw on
            v_x: Forward velocity (m/s)
            omega: Angular velocity (rad/s)
            max_v: Maximum linear velocity for scaling
            max_omega: Maximum angular velocity for scaling
        """
        h, w = image.shape[:2]

        # Bar dimensions
        bar_width = 200
        bar_height = 20
        bar_y = h - 80

        # --- Linear velocity bar (horizontal) ---
        # Background
        cv2.rectangle(
            image,
            (w // 2 - bar_width // 2, bar_y),
            (w // 2 + bar_width // 2, bar_y + bar_height),
            self.COLOR_WHITE,
            1
        )

        # Fill based on velocity
        fill_width = int((v_x / max_v) * (bar_width // 2))
        fill_color = self.COLOR_GREEN if v_x >= 0 else self.COLOR_RED

        if v_x >= 0:
            cv2.rectangle(
                image,
                (w // 2, bar_y + 2),
                (w // 2 + fill_width, bar_y + bar_height - 2),
                fill_color,
                -1
            )
        else:
            cv2.rectangle(
                image,
                (w // 2 + fill_width, bar_y + 2),
                (w // 2, bar_y + bar_height - 2),
                fill_color,
                -1
            )

        # Label
        cv2.putText(
            image, f"v_x: {v_x:+.2f} m/s",
            (w // 2 - bar_width // 2, bar_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            self.COLOR_WHITE, 1
        )

        # --- Angular velocity bar (horizontal) ---
        bar_y2 = h - 40

        # Background
        cv2.rectangle(
            image,
            (w // 2 - bar_width // 2, bar_y2),
            (w // 2 + bar_width // 2, bar_y2 + bar_height),
            self.COLOR_WHITE,
            1
        )

        # Fill based on velocity
        fill_width = int((omega / max_omega) * (bar_width // 2))
        fill_color = self.COLOR_BLUE if omega >= 0 else self.COLOR_ORANGE

        if omega >= 0:
            cv2.rectangle(
                image,
                (w // 2, bar_y2 + 2),
                (w // 2 + fill_width, bar_y2 + bar_height - 2),
                fill_color,
                -1
            )
        else:
            cv2.rectangle(
                image,
                (w // 2 + fill_width, bar_y2 + 2),
                (w // 2, bar_y2 + bar_height - 2),
                fill_color,
                -1
            )

        # Label
        cv2.putText(
            image, f"omega: {omega:+.2f} rad/s",
            (w // 2 - bar_width // 2, bar_y2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            self.COLOR_WHITE, 1
        )

    def _draw_fps(self, image: np.ndarray) -> None:
        """Draw FPS counter in top-right corner."""
        h, w = image.shape[:2]

        text = f"FPS: {self._current_fps:.1f}"
        cv2.putText(
            image, text,
            (w - 100, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            self.COLOR_WHITE, 1
        )

    def _draw_info(self, image: np.ndarray, info: dict) -> None:
        """
        Draw additional info in bottom-left corner.

        Args:
            image: Image to draw on
            info: Dictionary of info to display
        """
        y = image.shape[0] - 120

        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(
                image, text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                self.COLOR_WHITE, 1
            )
            y += 20

    def should_quit(self) -> bool:
        """
        Check if quit was requested (q or ESC pressed).

        Returns:
            True if quit requested
        """
        return self._last_key in [ord('q'), ord('Q'), 27]  # 27 = ESC

    def get_key(self) -> int:
        """
        Get the last key pressed.

        Returns:
            Key code, or -1 if no key pressed
        """
        return self._last_key

    def close(self) -> None:
        """Close the visualization window."""
        cv2.destroyWindow(self.window_name)
        cv2.waitKey(1)


def main():
    """
    Test the visualizer with a simulated feed.

    This demonstrates visualization features without
    requiring actual robot hardware.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from perception import Detection
    from state_machine import BehaviorState

    print("Visualizer Test")
    print("=" * 40)
    print("Press 'q' to quit")

    visualizer = Visualizer()

    # Simulate a moving detection
    frame_count = 0
    detection_x = 320  # Start at center

    try:
        while True:
            frame_count += 1

            # Create a test image
            image = np.zeros((480, 640, 3), dtype=np.uint8)

            # Add some visual elements to the background
            cv2.putText(
                image, "TEST MODE - No Camera",
                (200, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (100, 100, 100), 1
            )

            # Simulate detection moving back and forth
            detection_x = 320 + int(200 * np.sin(frame_count * 0.02))

            # Create mock detection
            detection = Detection.from_bbox(
                detection_x - 50, 190,
                detection_x + 50, 290,
                0.85
            )

            # Determine state based on position
            if abs(detection_x - 320) < 50:
                state = BehaviorState.TRACKING
            else:
                state = BehaviorState.TRACKING

            # Calculate mock velocities
            lateral_error = (detection_x - 320) / 640
            v_x = 0.2
            omega = -0.5 * lateral_error

            # Update visualization
            visualizer.update(
                image=image,
                detection=detection,
                state=state,
                v_x=v_x,
                omega=omega
            )

            if visualizer.should_quit():
                break

            time.sleep(0.033)  # ~30 FPS

    finally:
        visualizer.close()
        print("Visualization test complete")


if __name__ == "__main__":
    main()
