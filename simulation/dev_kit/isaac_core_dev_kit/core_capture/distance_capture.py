"""This file defines a class to capture distance from the isaac sim over ros2"""

# ==================== Imports ====================
import pickle

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Range

from .base_capture import BaseCapture


# ==================== Consts ====================
LASER_TOPIC_NAME = "/isaac_core/distance_sensor"


# ==================== The DistanceCapture class ====================
class DistanceCapture(Node, BaseCapture):
    """A class to capture distance from isaac sim over ros2"""

    def __init__(self, topic: str = LASER_TOPIC_NAME, name: str = "distance_capture_node") -> None:
        """Initialize the DistanceCapture class"""

        Node.__init__(self, node_name=name)
        BaseCapture.__init__(self)

        self.topic = topic
        self.is_capturing = False
        self.distances = {}
        self.cur_frame_id = 0

        self.isaac_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.sub = self.create_subscription(
            Range,
            self.topic,
            self.distance_callback,
            qos_profile=self.isaac_qos_profile
        )

    def start_capture(self) -> None:
        """Enable distance capturing"""

        self.is_capturing = True
        print("[DistanceCapture] distance capturing started.")

    def stop_capture(self) -> None:
        """Disable distance capturing"""

        self.is_capturing = False
        print("[DistanceCapture] distance capturing stopped.")

    def distance_callback(self, msg: Range) -> None:
        """Callback to handle incoming distance messages"""

        if not self.is_capturing:
            return

        self.distances[self.cur_frame_id] = msg

        self.cur_frame_id += 1


    def save_data_to(self, file_path: str) -> None:
        """Stops capturing and saves the distances pickle to the given file path"""

        if not self.distances:
            print("[DistanceCapture] No distances captured to save, skipping save.")
            return
        
        self.is_capturing = False

        with open(file_path, "wb") as f:
            pickle.dump(self.distances, f)

        print(f"[DistanceCapture] Saved captured distances to: {file_path}")

        self.distances = {}  # Clear distances after saving
        self.cur_frame_id = 0 # Reset frame id after saving
