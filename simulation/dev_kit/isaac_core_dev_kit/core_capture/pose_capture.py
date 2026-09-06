"""This file defines a class to capture pose from the isaac sim over ros2"""

# ==================== Imports ====================
import pickle

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geographic_msgs.msg import GeoPoseStamped

from .base_capture import BaseCapture


# ==================== Consts ====================
GLOBAL_POSE_TOPIC_NAME = "/isaac_core/global_pose"


# ==================== The PoseCapture class ====================
class PoseCapture(Node, BaseCapture):
    """A class to capture pose from isaac sim over ros2"""

    def __init__(self, topic: str = GLOBAL_POSE_TOPIC_NAME, name: str = "pose_capture_node") -> None:
        """Initialize the PoseCapture class"""

        Node.__init__(self, node_name=name)
        BaseCapture.__init__(self)

        self.topic = topic
        self.is_capturing = False
        self.poses = {}
        self.cur_frame_id = 0

        self.isaac_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.sub = self.create_subscription(
            GeoPoseStamped,
            self.topic,
            self.pose_callback,
            qos_profile=self.isaac_qos_profile
        )

    def start_capture(self) -> None:
        """Enable pose capturing"""

        self.is_capturing = True
        print("[PoseCapture] pose capturing started.")

    def stop_capture(self) -> None:
        """Disable pose capturing"""

        self.is_capturing = False
        print("[PoseCapture] pose capturing stopped.")

    def pose_callback(self, msg: GeoPoseStamped) -> None:
        """Callback to handle incoming pose messages"""

        if not self.is_capturing:
            return

        self.poses[self.cur_frame_id] = msg

        self.cur_frame_id += 1


    def save_data_to(self, file_path: str) -> None:
        """Stops capturing and saves the poses pickle to the given file path"""

        if not self.poses:
            print("[PoseCapture] No poses captured to save, skipping save.")
            return
        
        self.is_capturing = False

        with open(file_path, "wb") as f:
            pickle.dump(self.poses, f)

        print(f"[PoseCapture] Saved captured poses to: {file_path}")

        self.poses = {}  # Clear poses after saving
        self.cur_frame_id = 0 # Reset frame id after saving
