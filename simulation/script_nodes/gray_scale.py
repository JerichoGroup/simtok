"""Apply grayscale thermal brightness to configured prims based on distance and oscillation."""

from __future__ import annotations

import math
import random
import sys
import threading
from pathlib import Path
from typing import Optional

from std_msgs.msg import Empty
import rclpy
from isaac_ros2_messages.msg import FrameBboxes
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import omni.usd
from pxr import Gf, UsdGeom

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from config import get_config

    _cfg = get_config()
    _grayscale = _cfg.grayscale
    _oscillation_cfg = _cfg.grayscale_oscillation
    _prims_cfg = _cfg.grayscale_prims

    BBOX_TOPIC_NAME = _grayscale["bbox_topic_name"]
    TARGET_NAME = _grayscale["target_name"]
    RESET_OSCILLATION_TOPIC = _grayscale["reset_oscillation_topic"]
    NEW_OSCILLATION_TOPIC = _grayscale["new_oscillation_topic"]
    ALPHA = _grayscale["alpha"]

    OSCILLATION_MIN = _oscillation_cfg["min"]
    OSCILLATION_MAX = _oscillation_cfg["max"]
    OSCILLATION_NUM_VALUES_MIN = _oscillation_cfg["num_values_min"]
    OSCILLATION_NUM_VALUES_MAX = _oscillation_cfg["num_values_max"]

except Exception:
    # Fallback defaults when running inside Isaac Sim without access to the TOML
    BBOX_TOPIC_NAME = "/isaac_core/bbox"
    TARGET_NAME = "Cube"
    RESET_OSCILLATION_TOPIC = "/simtok/reset_oscillation"
    NEW_OSCILLATION_TOPIC = "/simtok/new_oscillation"
    ALPHA = 0.001

    OSCILLATION_MIN = -0.26
    OSCILLATION_MAX = 0.26
    OSCILLATION_NUM_VALUES_MIN = 300
    OSCILLATION_NUM_VALUES_MAX = 600

    _prims_cfg = [
        {"path": "/bboxes/Cube", "default_gray": 0.5},
        {"path": "/World/line2", "default_gray": 0.7},
    ]


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

        self.executor = SingleThreadedExecutor()

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
        """Update distance and advance the oscillation index on each bbox message."""
        self.current_distance = get_distance_to_target(msg, TARGET_NAME)

        if not self.oscillation_values:
            return

        if self.oscillation_index >= len(self.oscillation_values):
            self.oscillation_index = 0

        self.current_offset = self.oscillation_values[self.oscillation_index]
        self.oscillation_index += 1

    def restart_oscillation_profile_callback(self, _: Empty) -> None:
        """Reset the oscillation index to replay the current profile from the start."""
        self.oscillation_index = 0

        if self.oscillation_values:
            self.current_offset = self.oscillation_values[0]
        else:
            self.current_offset = 0.0

    def generate_new_oscillation_profile_callback(self, _: Empty) -> None:
        """Generate a new random oscillation profile and reset playback."""
        self.oscillation_values = generate_oscillation_values()
        self.oscillation_index = 0

        if self.oscillation_values:
            self.current_offset = self.oscillation_values[0]
        else:
            self.current_offset = 0.0

    def subscribe(self):
        """Subscribe to the bbox topic and start spinning in a background thread."""
        if self.bbox_subscriber is None:
            self.bbox_subscriber = self.node.create_subscription(
                FrameBboxes, BBOX_TOPIC_NAME, self.bbox_callback, self.qos_profile
            )

        if not self.is_spinning:
            threading.Thread(target=self._spin, daemon=True).start()
            self.is_spinning = True

    def _spin(self) -> None:
        """Run the ROS2 executor in a blocking loop."""
        self.executor.add_node(self.node)
        self.executor.spin()


class PrimColorController:
    """Control the grayscale color of a single USD prim via its bound material shader."""

    def __init__(self, prim_path: str, default_gray: float = 0.5):
        """Store the prim path and default gray value."""
        self.prim_path = prim_path
        self.default_gray = default_gray
        self.initialized = False
        self.base_gray = default_gray
        self.shader = None

    def _find_shader(self, stage):
        """Find the OmniPBR shader bound to this prim."""
        from pxr import UsdShade

        prim = stage.GetPrimAtPath(self.prim_path)
        if not prim.IsValid():
            return None

        binding_api = UsdShade.MaterialBindingAPI(prim)
        material = binding_api.GetDirectBinding().GetMaterial()

        if not material:
            return None

        for child in material.GetPrim().GetChildren():
            shader = UsdShade.Shader(child)
            if shader.GetIdAttr().Get() == "OmniPBR":
                return shader

        return None

    def initialize_base_gray(self, stage):
        """Find the shader and read its current diffuse color as the baseline."""
        from pxr import UsdShade

        self.shader = self._find_shader(stage)

        if self.shader is None:
            return

        diffuse_input = self.shader.GetInput("diffuse_color_constant")
        if diffuse_input:
            color = diffuse_input.Get()
            if color:
                self.base_gray = color[0]

        self.initialized = True

    def update(self, stage, offset: float, distance: Optional[float]):
        """Compute and apply the new grayscale value to the shader."""
        if not self.initialized:
            self.initialize_base_gray(stage)

        if self.shader is None:
            return

        gray = self.base_gray + offset
        gray = apply_thermal_fading(gray, distance)
        gray = max(0.0, min(1.0, gray))

        self.shader.GetInput("diffuse_color_constant").Set(Gf.Vec3f(gray, gray, gray))


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
        PrimColorController(prim_cfg["path"], prim_cfg.get("default_gray", 0.5))
        for prim_cfg in _prims_cfg
    ]


def compute(db):
    """Update all prim colors using the current oscillation offset and distance."""
    node = getattr(db.internal_state, "ros2_bbox_node", None)
    controllers = getattr(db.internal_state, "prim_controllers", None)

    if node is None or not controllers:
        return True

    try:
        stage = omni.usd.get_context().get_stage()

        print(
            f"Distance={node.current_distance if node.current_distance else 'N/A'} | "
            f"Offset={node.current_offset:+.4f}"
        )

        for controller in controllers:
            controller.update(stage, node.current_offset, node.current_distance)

    except Exception as e:
        print(f"Failed to update colors: {e}")

    return True


def cleanup(db):
    """Destroy the ROS2 node and shut down rclpy."""
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
    db.internal_state.prim_controllers = None
