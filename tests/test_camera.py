#!/usr/bin/env python3
"""
Camera Test Script
==================
This script tests the camera connection to Spot robot.
It captures a single image and saves it to disk.

Use this to verify:
- Network connection to Spot
- Camera access and image capture
- Image quality and format

Usage:
    python tests/test_camera.py
    python tests/test_camera.py --hostname 192.168.80.3 --save-path test_image.jpg

Author: Generated for Spot Person Follower Project
"""

import sys
import os
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spot_controller import SpotController


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test camera access on Spot robot"
    )

    parser.add_argument(
        "--hostname",
        type=str,
        default="192.168.80.3",
        help="Spot robot hostname/IP (default: 192.168.80.3)"
    )

    parser.add_argument(
        "--username",
        type=str,
        default="admin",
        help="Spot username (default: admin)"
    )

    parser.add_argument(
        "--password",
        type=str,
        default="",
        help="Spot password"
    )

    parser.add_argument(
        "--camera",
        type=str,
        default="frontleft_fisheye_image",
        choices=[
            "frontleft_fisheye_image",
            "frontright_fisheye_image",
            "left_fisheye_image",
            "right_fisheye_image",
            "back_fisheye_image"
        ],
        help="Camera source to use (default: frontleft_fisheye_image)"
    )

    parser.add_argument(
        "--save-path",
        type=str,
        default="test_capture.jpg",
        help="Path to save captured image (default: test_capture.jpg)"
    )

    parser.add_argument(
        "--display",
        action="store_true",
        help="Display the captured image in a window"
    )

    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Capture images continuously (press 'q' to quit)"
    )

    return parser.parse_args()


def test_single_capture(
    controller: SpotController,
    save_path: str,
    display: bool = False
) -> bool:
    """
    Capture and save a single image.

    Args:
        controller: Connected SpotController
        save_path: Path to save the image
        display: Whether to display the image

    Returns:
        True if successful
    """
    print("\nCapturing image...")

    start_time = time.time()
    image = controller.get_image()
    capture_time = time.time() - start_time

    if image is None:
        print("ERROR: Failed to capture image")
        return False

    print(f"Image captured successfully!")
    print(f"  Resolution: {image.shape[1]} x {image.shape[0]}")
    print(f"  Channels: {image.shape[2] if len(image.shape) > 2 else 1}")
    print(f"  Capture time: {capture_time*1000:.1f} ms")

    # Save image
    print(f"\nSaving image to: {save_path}")
    success = cv2.imwrite(save_path, image)

    if success:
        file_size = os.path.getsize(save_path)
        print(f"  File size: {file_size / 1024:.1f} KB")
    else:
        print("ERROR: Failed to save image")
        return False

    # Display if requested
    if display:
        print("\nDisplaying image (press any key to close)...")
        cv2.imshow("Spot Camera Test", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return True


def test_continuous_capture(
    controller: SpotController,
    display: bool = True
) -> None:
    """
    Capture images continuously.

    Args:
        controller: Connected SpotController
        display: Whether to display images (required for continuous mode)
    """
    print("\nStarting continuous capture...")
    print("Press 'q' to quit, 's' to save current frame")

    frame_count = 0
    fps_start = time.time()
    current_fps = 0.0

    while True:
        # Capture frame
        image = controller.get_image()

        if image is None:
            print("Warning: Failed to capture frame")
            time.sleep(0.1)
            continue

        frame_count += 1

        # Calculate FPS
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            current_fps = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # Add FPS overlay
        cv2.putText(
            image, f"FPS: {current_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 255, 0), 2
        )

        # Display
        cv2.imshow("Spot Camera - Continuous", image)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"capture_{int(time.time())}.jpg"
            cv2.imwrite(filename, image)
            print(f"Saved: {filename}")

    cv2.destroyAllWindows()
    print("Continuous capture ended")


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Spot Camera Test")
    print("=" * 60)
    print(f"Hostname: {args.hostname}")
    print(f"Camera: {args.camera}")
    print("=" * 60)

    # Check password
    if not args.password:
        print("\nWARNING: No password provided. You may need to specify --password")

    # Create controller
    controller = SpotController(
        hostname=args.hostname,
        username=args.username,
        password=args.password,
        camera_source=args.camera
    )

    try:
        # Connect to robot
        print("\nConnecting to Spot...")
        controller.connect()
        print("Connected!")

        # Get robot state
        state = controller.get_robot_state()
        print(f"\nRobot State:")
        print(f"  Battery: {state.get('battery_percentage', 'N/A')}%")
        print(f"  E-Stop engaged: {state.get('estop_engaged', 'N/A')}")

        # Run appropriate test
        if args.continuous:
            test_continuous_capture(controller, display=True)
        else:
            success = test_single_capture(
                controller,
                save_path=args.save_path,
                display=args.display
            )

            if success:
                print("\n✓ Camera test PASSED")
            else:
                print("\n✗ Camera test FAILED")
                sys.exit(1)

    except KeyboardInterrupt:
        print("\nTest interrupted by user")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Disconnect
        print("\nDisconnecting from Spot...")
        controller.disconnect()
        print("Disconnected")


if __name__ == "__main__":
    main()
