"""This file implements a script node that subscribes to a ROS2 bbox topic,
computes the 3D distance to a target, and applies grayscale brightness
using an in-memory oscillation profile to the cube and line2 prims."""

from __future__ import annotations

import math
from std_msgs.msg import Empty
import random
import rclpy
import threading
from rclpy.node import Node
from isaac_ros2_messages.msg import FrameBboxes
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from typing import Optional
import omni.usd
from pxr import Gf, UsdGeom


BBOX_TOPIC_NAME = "/isaac_core/bbox"
TARGET_NAME = "Cube"
CUBE_PRIM_PATH = "/bboxes/Cube"
LINE_PRIM_PATH = "/World/line2"

OSCILLATION_MIN = -0.26
OSCILLATION_MAX = 0.26
OSCILLATION_NUM_VALUES_MIN = 300
OSCILLATION_NUM_VALUES_MAX = 600

RESET_OSCILLATION_TOPIC = "/simtok/reset_oscillation"
NEW_OSCILLATION_TOPIC = "/simtok/new_oscillation"

ALPHA = 0.001  # thermal fading coefficient


def get_distance_to_target(msg: FrameBboxes, target_name: str = TARGET_NAME) -> Optional[float]:
    """Return 3D Euclidean distance to the target bbox."""
    for bbox in msg.bboxes:
        if bbox.target_name == target_name:
            return math.sqrt(
                bbox.distance_x**2 +
                bbox.distance_y**2 +
                bbox.distance_z**2
            )
    return None


def apply_thermal_fading(gray_value: float, distance: Optional[float], alpha: float = ALPHA) -> float:
    """Apply exponential thermal fading."""
    if distance is None:
        return gray_value
    return gray_value * math.exp(-alpha * distance)


def generate_oscillation_values() -> list[float]:
    num_values = random.randint(
        OSCILLATION_NUM_VALUES_MIN,
        OSCILLATION_NUM_VALUES_MAX,
    )

    values = sorted(
        random.uniform(OSCILLATION_MIN, OSCILLATION_MAX)
        for _ in range(num_values)
    )

    return values + values[::-1]


class ROS2BboxNode:
    """Subscribes to bbox topic and produces distance + oscillation offset."""

    def __init__(self):
        self.node = rclpy.create_node("ros2_gray_scale_node")
        self.is_spinning = False
        self.bbox_subscriber = None

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.control_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.executor = MultiThreadedExecutor()

        try:
            self.node.declare_parameter("use_sim_time", True)
        except rclpy.exceptions.ParameterAlreadyDeclaredException:
            pass

        self.current_distance: Optional[float] = None
        self.current_offset: float = 0.0

        self.oscillation_values = generate_oscillation_values()
        self.oscillation_index = 0

        self.restart_profile_subscriber = self.node.create_subscription(
            Empty,
            RESET_OSCILLATION_TOPIC,
            self.restart_oscillation_profile_callback,
            self.control_qos,
        )

        self.generate_profile_subscriber = self.node.create_subscription(
            Empty,
            NEW_OSCILLATION_TOPIC,
            self.generate_new_oscillation_profile_callback,
            self.control_qos,
        )


    def bbox_callback(self, msg: FrameBboxes) -> None:
        self.current_distance = get_distance_to_target(msg, TARGET_NAME)

        if not self.oscillation_values:
            return

        if self.oscillation_index >= len(self.oscillation_values):
            self.oscillation_index = 0

        self.current_offset = self.oscillation_values[self.oscillation_index]
        self.oscillation_index += 1


    def restart_oscillation_profile_callback(self, _: Empty) -> None:
        """
        Restart playback from the beginning of the current oscillation profile.
        """

        self.oscillation_index = 0

        if self.oscillation_values:
            self.current_offset = self.oscillation_values[0]
        else:
            self.current_offset = 0.0


    def generate_new_oscillation_profile_callback(self, _: Empty) -> None:
        """
        Generate a new random oscillation profile and restart playback.
        """

        self.oscillation_values = generate_oscillation_values()
        self.oscillation_index = 0

        if self.oscillation_values:
            self.current_offset = self.oscillation_values[0]
        else:
            self.current_offset = 0.0


    def subscribe(self):
        """
        Subscribe to bbox topic.
        """

        if self.bbox_subscriber is None:
            self.bbox_subscriber = self.node.create_subscription(
                FrameBboxes, BBOX_TOPIC_NAME, self.bbox_callback, self.qos_profile
            )

        if not self.is_spinning:
            threading.Thread(target=self._spin, daemon=True).start()
            self.is_spinning = True

    def _spin(self) -> None:
        self.executor.add_node(self.node)
        self.executor.spin()


class CubeColorController:
    """Handles grayscale updates for the cube prim."""

    def __init__(self, prim_path: str, default_gray: float = 0.5):
        self.prim_path = prim_path
        self.default_gray = default_gray
        self.initialized = False
        self.base_gray = default_gray

    def initialize_base_gray(self, stage):
        """
        Initialize the base gray value from the cube prim's display color.
        """

        prim = stage.GetPrimAtPath(self.prim_path)

        if not prim.IsValid():
            return

        mesh = UsdGeom.Mesh(prim)
        attr = mesh.CreateDisplayColorAttr()
        colors = attr.Get()

        self.base_gray = colors[0][0] if colors else self.default_gray

        self.initialized = True

    def update(self, stage, offset: float, distance: Optional[float]):
        """
        Update the cube prim's color based on the current offset and distance.
        """

        prim = stage.GetPrimAtPath(self.prim_path)

        if not prim.IsValid():
            return

        if not self.initialized:
            self.initialize_base_gray(stage)

        gray = self.base_gray + offset
        gray = apply_thermal_fading(gray, distance)
        gray = max(0.0, min(1.0, gray))

        mesh = UsdGeom.Mesh(prim)
        attr = mesh.CreateDisplayColorAttr()
        attr.Set([Gf.Vec3f(gray, gray, gray)])


class LineColorController:
    """Handles grayscale updates for the line2 prim."""

    def __init__(self, prim_path: str, default_gray: float = 0.7):
        self.prim_path = prim_path
        self.default_gray = default_gray
        self.initialized = False
        self.base_gray = default_gray

    def initialize_base_gray(self, stage):
        """
        Initialize the base gray value from the line2 prim's display color.
        """

        prim = stage.GetPrimAtPath(self.prim_path)

        if not prim.IsValid():
            return

        mesh = UsdGeom.Mesh(prim)
        attr = mesh.CreateDisplayColorAttr()
        colors = attr.Get()

        self.base_gray = colors[0][0] if colors else self.default_gray

        self.initialized = True

    def update(self, stage, offset: float, distance: Optional[float]):
        """
        Update the line2 prim's color based on the current offset and distance.
        """

        prim = stage.GetPrimAtPath(self.prim_path)

        if not prim.IsValid():
            return

        if not self.initialized:
            self.initialize_base_gray(stage)

        gray = self.base_gray + offset
        gray = apply_thermal_fading(gray, distance)
        gray = max(0.0, min(1.0, gray))

        mesh = UsdGeom.Mesh(prim)
        attr = mesh.CreateDisplayColorAttr()
        attr.Set([Gf.Vec3f(gray, gray, gray)])


def setup(db):
    if not rclpy.ok():
        try:
            rclpy.init()
        except Exception as e:
            raise RuntimeError("Failed to initialize rclpy") from e

    db.internal_state.ros2_bbox_node = ROS2BboxNode()
    db.internal_state.ros2_bbox_node.subscribe()

    db.internal_state.cube_controller = CubeColorController(CUBE_PRIM_PATH, default_gray=0.5)
    db.internal_state.line_controller = LineColorController(LINE_PRIM_PATH, default_gray=0.7)


def compute(db):
    """Apply grayscale updates to cube and line2 using shared oscillation."""

    node = getattr(db.internal_state, "ros2_bbox_node", None)
    cube_controller = getattr(db.internal_state, "cube_controller", None)
    line_controller = getattr(db.internal_state, "line_controller", None)

    if node is None or cube_controller is None or line_controller is None:
        return True

    try:
        stage = omni.usd.get_context().get_stage()

        print(
            f"Distance={node.current_distance if node.current_distance else 'N/A'} | "
            f"Offset={node.current_offset:+.4f}"
        )

        # Both use the SAME offset and SAME fading
        cube_controller.update(stage, node.current_offset, node.current_distance)
        line_controller.update(stage, node.current_offset, node.current_distance)

    except Exception as e:
        print(f"Failed to update colors: {e}")

    return True


def cleanup(db):
    node = getattr(db.internal_state, "ros2_bbox_node", None)

    if node is not None:
        try:
            node.node.destroy_node()
        except Exception as e:
            print(e)

    if rclpy.ok():
        try:
            rclpy.shutdown()
        except Exception as e:
            print(e)

    db.internal_state.ros2_bbox_node = None
    db.internal_state.cube_controller = None
    db.internal_state.line_controller = None
