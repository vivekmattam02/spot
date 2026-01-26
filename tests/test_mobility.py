#!/usr/bin/env python3
"""
Mobility Test Script
====================
This script tests basic mobility commands on the Spot robot.
It makes Spot stand, move forward briefly, and sit.

WARNING: This script will make the robot move!
         Ensure the area is clear before running.

Use this to verify:
- Motor power control
- Stand/sit commands
- Basic velocity commands
- E-Stop integration

Usage:
    python tests/test_mobility.py
    python tests/test_mobility.py --hostname 192.168.80.3
    python tests/test_mobility.py --velocity 0.3 --duration 1.0

Author: Generated for Spot Person Follower Project
"""

import sys
import os
import argparse
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spot_controller import SpotController


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test basic mobility on Spot robot"
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
        "--velocity",
        type=float,
        default=0.5,
        help="Forward velocity for test (m/s, default: 0.5)"
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Duration of forward motion (seconds, default: 2.0)"
    )

    parser.add_argument(
        "--skip-motion",
        action="store_true",
        help="Skip the motion test (only test stand/sit)"
    )

    parser.add_argument(
        "--test-rotation",
        action="store_true",
        help="Also test rotation (turn left then right)"
    )

    return parser.parse_args()


def confirm_action(message: str) -> bool:
    """
    Ask user to confirm before proceeding.

    Args:
        message: Confirmation message

    Returns:
        True if user confirms
    """
    print(f"\n{message}")
    response = input("Type 'yes' to confirm, anything else to cancel: ")
    return response.lower() == 'yes'


def test_stand_sit(controller: SpotController) -> bool:
    """
    Test stand and sit commands.

    Args:
        controller: Connected SpotController

    Returns:
        True if successful
    """
    print("\n" + "=" * 40)
    print("TEST: Stand and Sit")
    print("=" * 40)

    try:
        # Stand
        print("\nCommanding robot to STAND...")
        controller.stand()
        print("✓ Robot is standing")

        # Wait a moment
        print("Waiting 3 seconds...")
        time.sleep(3.0)

        # Sit
        print("\nCommanding robot to SIT...")
        controller.sit()
        print("✓ Robot is sitting")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False


def test_forward_motion(
    controller: SpotController,
    velocity: float,
    duration: float
) -> bool:
    """
    Test forward motion.

    Args:
        controller: Connected SpotController
        velocity: Forward velocity (m/s)
        duration: Duration of motion (s)

    Returns:
        True if successful
    """
    print("\n" + "=" * 40)
    print("TEST: Forward Motion")
    print("=" * 40)
    print(f"Velocity: {velocity} m/s")
    print(f"Duration: {duration} s")
    print(f"Expected distance: {velocity * duration:.2f} m")

    try:
        # Stand first
        print("\nStanding up...")
        controller.stand()
        print("✓ Standing")

        time.sleep(1.0)

        # Start motion
        print(f"\nMoving forward at {velocity} m/s...")

        # Send commands at 10 Hz for smooth motion
        control_rate = 10.0
        control_period = 1.0 / control_rate
        num_cycles = int(duration * control_rate)

        for i in range(num_cycles):
            # Check E-Stop
            if not controller.check_estop():
                print("WARNING: E-Stop engaged, stopping")
                break

            controller.move(v_x=velocity, v_y=0.0, omega=0.0)
            time.sleep(control_period)

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Progress: {(i + 1) / num_cycles * 100:.0f}%")

        # Stop
        print("\nStopping...")
        controller.stop()
        print("✓ Motion complete")

        time.sleep(1.0)

        # Sit
        print("\nSitting down...")
        controller.sit()
        print("✓ Sitting")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        # Try to stop and sit
        try:
            controller.stop()
            controller.sit()
        except:
            pass
        return False


def test_rotation(controller: SpotController) -> bool:
    """
    Test rotation (turning).

    Args:
        controller: Connected SpotController

    Returns:
        True if successful
    """
    print("\n" + "=" * 40)
    print("TEST: Rotation")
    print("=" * 40)

    rotation_speed = 0.3  # rad/s
    rotation_duration = 2.0  # seconds

    try:
        # Stand first
        print("\nStanding up...")
        controller.stand()
        print("✓ Standing")

        time.sleep(1.0)

        # Turn left
        print(f"\nTurning LEFT at {rotation_speed} rad/s for {rotation_duration}s...")

        control_rate = 10.0
        control_period = 1.0 / control_rate
        num_cycles = int(rotation_duration * control_rate)

        for i in range(num_cycles):
            controller.move(v_x=0.0, v_y=0.0, omega=rotation_speed)
            time.sleep(control_period)

        controller.stop()
        print("✓ Left turn complete")

        time.sleep(1.0)

        # Turn right
        print(f"\nTurning RIGHT at {rotation_speed} rad/s for {rotation_duration}s...")

        for i in range(num_cycles):
            controller.move(v_x=0.0, v_y=0.0, omega=-rotation_speed)
            time.sleep(control_period)

        controller.stop()
        print("✓ Right turn complete")

        time.sleep(1.0)

        # Sit
        print("\nSitting down...")
        controller.sit()
        print("✓ Sitting")

        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        try:
            controller.stop()
            controller.sit()
        except:
            pass
        return False


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("Spot Mobility Test")
    print("=" * 60)
    print(f"Hostname: {args.hostname}")
    print(f"Test velocity: {args.velocity} m/s")
    print(f"Test duration: {args.duration} s")
    print("=" * 60)

    # Safety warnings
    print("\n" + "!" * 60)
    print("WARNING: This test will make the robot MOVE!")
    print("Ensure the area around the robot is CLEAR.")
    print("Be ready to press the E-Stop if needed.")
    print("!" * 60)

    # Confirm before proceeding
    if not confirm_action("Do you want to proceed with the mobility test?"):
        print("Test cancelled by user")
        return

    # Check password
    if not args.password:
        print("\nWARNING: No password provided. You may need to specify --password")

    # Create controller
    controller = SpotController(
        hostname=args.hostname,
        username=args.username,
        password=args.password
    )

    test_results = {}

    try:
        # Connect
        print("\nConnecting to Spot...")
        controller.connect()
        print("✓ Connected")

        # Get robot state
        state = controller.get_robot_state()
        print(f"\nRobot State:")
        print(f"  Battery: {state.get('battery_percentage', 'N/A')}%")
        print(f"  E-Stop engaged: {state.get('estop_engaged', 'N/A')}")

        # Check E-Stop
        if not controller.check_estop():
            print("\n✗ E-Stop is engaged! Release E-Stop before testing.")
            return

        # Test 1: Stand and Sit
        test_results['stand_sit'] = test_stand_sit(controller)

        if not args.skip_motion:
            # Confirm before motion test
            print("\n" + "-" * 40)
            if not confirm_action(
                f"Ready to test forward motion ({args.velocity} m/s for {args.duration}s). "
                "Ensure area is clear."
            ):
                print("Motion test skipped")
                test_results['forward_motion'] = None
            else:
                # Test 2: Forward Motion
                test_results['forward_motion'] = test_forward_motion(
                    controller,
                    velocity=args.velocity,
                    duration=args.duration
                )

            # Test 3: Rotation (optional)
            if args.test_rotation:
                print("\n" + "-" * 40)
                if not confirm_action(
                    "Ready to test rotation. Ensure area is clear."
                ):
                    print("Rotation test skipped")
                    test_results['rotation'] = None
                else:
                    test_results['rotation'] = test_rotation(controller)

        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        all_passed = True
        for test_name, result in test_results.items():
            if result is None:
                status = "SKIPPED"
            elif result:
                status = "✓ PASSED"
            else:
                status = "✗ FAILED"
                all_passed = False

            print(f"  {test_name}: {status}")

        print("=" * 60)

        if all_passed:
            print("\n✓ All mobility tests PASSED")
        else:
            print("\n✗ Some tests FAILED")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user!")
        print("Attempting to stop and sit robot...")
        try:
            controller.stop()
            time.sleep(0.5)
            controller.sit()
        except:
            pass

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

        # Try to stop robot
        print("\nAttempting emergency stop...")
        try:
            controller.stop()
            time.sleep(0.5)
            controller.sit()
        except:
            pass

        sys.exit(1)

    finally:
        # Disconnect
        print("\nDisconnecting from Spot...")
        try:
            controller.disconnect()
        except:
            pass
        print("Disconnected")


if __name__ == "__main__":
    main()
