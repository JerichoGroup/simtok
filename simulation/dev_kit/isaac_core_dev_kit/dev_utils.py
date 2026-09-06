"""This module contains utility functions for development in Isaac Core."""

# ==================== Imports ====================
import time
from pathlib import Path

import rclpy

from isaac_ros2_messages.msg import Gimbal, SATOutput


# ==================== delete cesium cache ====================
def delete_cesium_cache() -> None:
    """Trying to delete the cesium cache file. at ~/.cache/ov/cesium-request-cache.sqlite-wal"""

    cesium_cache_abs_path = Path.home() / ".cache" / "ov" / "cesium-request-cache.sqlite-wal"

    print("Trying to delete cesium cache...")

    if cesium_cache_abs_path.exists():
        cesium_cache_abs_path.unlink(missing_ok=True)
        print(f"Deleted Cesium cache at: {cesium_cache_abs_path}")
    else:
        print(f"No Cesium cache found at: {cesium_cache_abs_path}")


# ==================== safe rclpy init ====================
def safe_rclpy_init() -> None:
    """Safely initialize rclpy if not already initialized."""

    if not rclpy.ok():
        rclpy.init()


# ==================== safe rclpy shutdown ====================
def safe_rclpy_shutdown() -> None:
    """Safely shutdown rclpy if not already shutdown."""

    if rclpy.ok():
        rclpy.shutdown()


# ==================== set gimbal angle ====================
def set_gimbal_angle(roll: float, pitch: float, yaw: float, times: int = 3, shutdown_rclpy: bool = False) -> None:
    """Set the gimbal angle by publishing to the /isaac_core/gimbal topic."""

    safe_rclpy_init()

    gimbal_node = rclpy.create_node("gimbal_control_node")
    pub = gimbal_node.create_publisher(Gimbal, "/isaac_core/gimbal", 10)

    msg = Gimbal()
    msg.roll = roll
    msg.pitch = pitch
    msg.yaw = yaw

    for _ in range(times):
        pub.publish(msg)
        print(f"Sent gimbal msg: {msg}")
        time.sleep(0.05)

    gimbal_node.destroy_node()
    if shutdown_rclpy:
        safe_rclpy_shutdown()


# ==================== save_current_frame_to ====================
def save_current_frame_to(image_path: str, shutdown_rclpy: bool = False) -> None:
    """Save the current camera POV frame to the given image path (should end with .png), by publishing to the /isaac_core/sat topic."""

    safe_rclpy_init()

    sat_node = rclpy.create_node("save_sat_node")
    pub = sat_node.create_publisher(SATOutput, "/isaac_core/sat", 10)

    msg = SATOutput()
    msg.output_path = image_path

    pub.publish(msg)
    print(f"Sent SATOutput msg: {msg}")
    
    sat_node.destroy_node()
    if shutdown_rclpy:
        safe_rclpy_shutdown()
