"""this file implements a script node that subscribes to a ros2 bbox topic,
computes 3D distance to a target, and applies grayscale brightness
from an oscillation file to the target prim"""

# ==================== imports ====================
from __future__ import annotations

import os
import math
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


# ==================== consts ====================
BBOX_TOPIC_NAME = "/isaac_core/bbox"
TARGET_NAME = "Cube"
PRIM_PATH = "/bboxes/Cube"

# RGB oscillation consts
OSCILLATION_MIN = -0.2
OSCILLATION_MAX = 0.2
OSCILLATION_NUM_VALUES = 250  # number of oscillation values to generate
PROJECT_ROOT = os.getcwd()

OSCILLATION_FILE_PATH = os.path.join(
    PROJECT_ROOT,
    "tmp",
    "rgb_oscillation.txt",
)

# ==================== distance calculation ====================
def get_distance_to_target(msg: FrameBboxes, target_name: str = TARGET_NAME) -> Optional[float]:
    """
    Search through bboxes in the FrameBboxes message for TARGET_NAME
    and compute the 3D Euclidean distance from the camera using distance_x/y/z.

    The 3D distance formula: d = sqrt(dx^2 + dy^2 + dz^2)
    """

    for bbox in msg.bboxes:
        if bbox.target_name == target_name:
            distance = math.sqrt(
                bbox.distance_x ** 2 +
                bbox.distance_y ** 2 +
                bbox.distance_z ** 2
            )
            return distance
    return None


# ==================== RGB oscillation - write ====================
def create_rgb_oscillation(file_path: str = OSCILLATION_FILE_PATH,
                           min_val: float = OSCILLATION_MIN,
                           max_val: float = OSCILLATION_MAX,
                           num_values: int = OSCILLATION_NUM_VALUES) -> None:
    """
    Generate random grayscale values between min_val and max_val
    and write them to a text file. Each line contains one float value.

    These values represent gray brightness levels (0.0 = black, 1.0 = white).
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    values = [
        random.uniform(min_val, max_val)
        for _ in range(num_values)
    ]

    values.sort()
    values = values + values[::-1]

    with open(file_path, "w") as f:
        for value in values:
            f.write(f"{value:.6f}\n")


# ==================== RGB oscillation - read ====================
def load_oscillation_values(file_path: str = OSCILLATION_FILE_PATH) -> list:
    """
    Load all oscillation values from the file once.

    Returns:
        List of grayscale brightness floats [0.0-1.0], or empty list if file doesn't exist
    """
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, 'r') as f:
            return [float(line.strip()) for line in f.readlines() if line.strip()]
    except (IOError, ValueError):
        return []


# ==================== the ROS2BboxNode class ====================
class ROS2BboxNode:
    """this class subscribes to the bbox topic and computes distance"""

    def __init__(self):
        """initialize the node and subscriber"""

        self.node = rclpy.create_node("ros2_gray_scale_node")
        self.is_spinning = False
        self.bbox_subscriber = None

        self.qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        try:
            self.node.declare_parameter("use_sim_time", True)
        except rclpy.exceptions.ParameterAlreadyDeclaredException:
            pass

        self.current_distance = None
        self.current_gray = None
        self.current_offset = 0.0

        # Generate oscillation file on init if it doesn't exist
        if not os.path.exists(OSCILLATION_FILE_PATH):
            create_rgb_oscillation()

        # Load oscillation values once
        self.oscillation_values = load_oscillation_values()
        self.oscillation_index = 0


    def bbox_callback(self, msg: FrameBboxes) -> None:
        """callback for BBOXOutput messages - compute distance and cycle gray value"""

        self.current_distance = get_distance_to_target(msg, TARGET_NAME)

        # Cycle through preloaded grayscale values
        if self.oscillation_values:
            self.current_offset = self.oscillation_values[
                self.oscillation_index % len(self.oscillation_values)
            ]
            self.oscillation_index += 1


    def subscribe(self):
        """subscribe to the specified topic"""

        if self.bbox_subscriber is None:
            self.bbox_subscriber = self.node.create_subscription(
                FrameBboxes, BBOX_TOPIC_NAME, self.bbox_callback, self.qos_profile
            )

        if not self.is_spinning:
            threading.Thread(target=spin_node, args=(self.node,), daemon=True).start()
            self.is_spinning = True


# ==================== Helper - spin node ====================
def spin_node(node: Node) -> None:
    """spin the given node"""

    executor = MultiThreadedExecutor()
    executor.add_node(node)
    executor.spin()


# ==================== script Node Functions ====================

# ==================== setup
def setup(db):
    """Initialize camera info"""

    if not rclpy.ok():
        try:
            rclpy.init()
        except Exception as e:
            raise RuntimeError("Failed to initialize rclpy") from e

    db.internal_state.ros2_bbox_node = ROS2BboxNode()
    db.internal_state.ros2_bbox_node.subscribe()


# ==================== compute
def compute(db):
    """Apply the grayscale value with oscillation offset to the cube."""

    node = getattr(db.internal_state, "ros2_bbox_node", None)
    if node is None:
        return True

    try:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(PRIM_PATH)

        if not prim.IsValid():
            return True

        mesh = UsdGeom.Mesh(prim)
        attr = mesh.CreateDisplayColorAttr()

        if node.current_gray is None:   
            colors = attr.Get() 
            node.current_gray = colors[0][0] if colors else 0.5 

        gray_value = node.current_gray + node.current_offset
        gray_value = max(0.0, min(1.0, gray_value))

        attr.Set([
            Gf.Vec3f(
                gray_value,
                gray_value,
                gray_value
            )
        ])

    except Exception as e:
        print(f"Failed to update cube color: {e}")
    return True


# ==================== cleanup
def cleanup(db):
    """Cleanup ROS resources."""

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