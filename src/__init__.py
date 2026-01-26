"""
Spot Person Follower - Source Package
=====================================
Main source modules for the person-following system.
"""

from .spot_controller import SpotController
from .perception import PersonDetector, Detection
from .visual_servoing import VisualServoingController, ControlOutput
from .state_machine import BehaviorStateMachine, BehaviorState

__all__ = [
    "SpotController",
    "PersonDetector",
    "Detection",
    "VisualServoingController",
    "ControlOutput",
    "BehaviorStateMachine",
    "BehaviorState",
]
