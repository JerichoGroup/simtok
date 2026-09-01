"""Apply grayscale thermal brightness to configured prims based on distance and oscillation."""

from __future__ import annotations

import logging
import math
import random
import threading
from typing import Optional

logger = logging.getLogger(__name__)

from std_msgs.msg import Empty
import rclpy
from isaac_ros2_messages.msg import FrameBboxes
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import omni.usd
from pxr import Gf, UsdGeom

from config import get_config

_config = get_config()
_grayscale = _config.grayscale
_oscillation_config = _config.grayscale_oscillation
_prims_config = _config.grayscale_prims

BBOX_TOPIC_NAME = _grayscale["bbox_topic_name"]
TARGET_NAME = _grayscale["target_name"]
RESET_OSCILLATION_TOPIC = _grayscale["reset_oscillation_topic"]
NEW_OSCILLATION_TOPIC = _grayscale["new_oscillation_topic"]
ALPHA = _grayscale["alpha"]

OSCILLATION_MIN = _oscillation_config["min"]
OSCILLATION_MAX = _oscillation_config["max"]
OSCILLATION_NUM_VALUES_MIN = _oscillation_config["num_values_min"]
OSCILLATION_NUM_VALUES_MAX = _oscillation_config["num_values_max"]


def get_distance_to_target(msg: FrameBboxes, target_name: str = TARGET_NAME) -> Optional[float]:
    """Return the 3D Euclidean distance to the named target from a bbox message."""
    for bbox in msg.bboxes:
        if bbox.target_name == target_name:
            return math.sqrt(
                bbox.distance_x**2 +
                bbox.distance_y**2 +
                bbox.distance_z**2
            )
    return None


def apply_thermal_fading(gray_value: float, distance: Optional[float], alpha: float = ALPHA) -> float:
    """Apply exponential thermal attenuation based on distance."""
    if distance is None:
        return gray_value
    return gray_value * math.exp(-alpha * distance)


def generate_oscillation_values() -> list[float]:
    """Generate a symmetric random oscillation profile."""
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
    """Subscribe to bbox and oscillation control topics to produce distance and offset."""

    def __init__(self):
        """Create the ROS2 node with bbox and control subscriptions."""
        self.node = rclpy.create_node("ros2_gray_scale_node")
        self.is_spinning = False
        self.bbox_subscriber = None

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
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

        self._lock = threading.Lock()
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
        """Update distance and advance the oscillation index on each bbox message."""
        distance = get_distance_to_target(msg, TARGET_NAME)

        with self._lock:
            self.current_distance = distance

            if not self.oscillation_values:
                return

            if self.oscillation_index >= len(self.oscillation_values):
                self.oscillation_index = 0

            self.current_offset = self.oscillation_values[self.oscillation_index]
            self.oscillation_index += 1

    def restart_oscillation_profile_callback(self, _: Empty) -> None:
        """Reset the oscillation index to replay the current profile from the start."""
        with self._lock:
            self.oscillation_index = 0
            self.current_offset = self.oscillation_values[0] if self.oscillation_values else 0.0

    def generate_new_oscillation_profile_callback(self, _: Empty) -> None:
        """Generate a new random oscillation profile and reset playback."""
        new_values = generate_oscillation_values()

        with self._lock:
            self.oscillation_values = new_values
            self.oscillation_index = 0
            self.current_offset = new_values[0] if new_values else 0.0

    def get_state(self) -> tuple[Optional[float], float]:
        """Return (distance, offset) as a consistent snapshot."""
        with self._lock:
            return self.current_distance, self.current_offset

    def subscribe(self):
        """Subscribe to the bbox topic and start spinning in a background thread."""
        if self.bbox_subscriber is None:
            self.bbox_subscriber = self.node.create_subscription(
                FrameBboxes, BBOX_TOPIC_NAME, self.bbox_callback, self.qos_profile
            )

        if not self.is_spinning:
            self.executor.add_node(self.node)
            threading.Thread(target=self.executor.spin, daemon=True).start()
            self.is_spinning = True


class PrimColorController:
    """Control the grayscale display color of a single USD prim."""

    def __init__(self, prim_path: str, default_gray: float = 0.5):
        """Store the prim path and default gray value."""
        self.prim_path = prim_path
        self.default_gray = default_gray
        self.initialized = False
        self.base_gray = default_gray

    def initialize_base_gray(self, stage):
        """Read the prim's current display color to establish the base gray."""
        prim = stage.GetPrimAtPath(self.prim_path)

        if not prim.IsValid():
            return

        mesh = UsdGeom.Mesh(prim)
        attr = mesh.CreateDisplayColorAttr()
        colors = attr.Get()

        self.base_gray = colors[0][0] if colors else self.default_gray
        self.initialized = True

    def update(self, stage, offset: float, distance: Optional[float]):
        """Compute and apply the new grayscale value to the prim."""
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
    """Initialize rclpy, the bbox node, and color controllers for all configured prims."""
    if not rclpy.ok():
        try:
            rclpy.init()
        except Exception as e:
            raise RuntimeError("Failed to initialize rclpy") from e

    db.internal_state.ros2_bbox_node = ROS2BboxNode()
    db.internal_state.ros2_bbox_node.subscribe()

    db.internal_state.prim_controllers = [
        PrimColorController(prim_config["path"], prim_config.get("default_gray", 0.5))
        for prim_config in _prims_config
    ]


def compute(db):
    """Update all prim colors using the current oscillation offset and distance."""
    node = getattr(db.internal_state, "ros2_bbox_node", None)
    controllers = getattr(db.internal_state, "prim_controllers", None)

    if node is None or not controllers:
        return True

    try:
        stage = omni.usd.get_context().get_stage()

        distance, offset = node.get_state()

        logger.info(
            "Distance=%s | Offset=%+.4f",
            distance if distance else "N/A",
            offset,
        )

        for controller in controllers:
            controller.update(stage, offset, distance)

    except Exception as e:
        logger.error("Failed to update colors: %s", e)

    return True


def cleanup(db):
    """Destroy the ROS2 node and shut down rclpy."""
    node = getattr(db.internal_state, "ros2_bbox_node", None)

    if node is not None:
        try:
            node.node.destroy_node()
        except Exception as e:
            logger.error("Error destroying ROS2 node: %s", e)

    db.internal_state.ros2_bbox_node = None
    db.internal_state.prim_controllers = None
