"""This file defines a class to capture bbox from the isaac sim over ros2"""

# ==================== Imports ====================
import pickle

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from isaac_ros2_messages.msg import FrameBboxes

from .base_capture import BaseCapture


# ==================== Consts ====================
BBOXES_TOPIC_NAME = "/isaac_core/bbox"


# ==================== The BboxCapture class ====================
class BboxCapture(Node, BaseCapture):
    """A class to capture bbox from isaac sim over ros2"""

    def __init__(self, topic: str = BBOXES_TOPIC_NAME, name: str = "bbox_capture_node") -> None:
        """Initialize the BboxCapture class"""

        Node.__init__(self, node_name=name)
        BaseCapture.__init__(self)

        self.topic = topic
        self.is_capturing = False
        self.bboxes = {}
        self.cur_frame_id = 0

        self.isaac_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.sub = self.create_subscription(
            FrameBboxes,
            self.topic,
            self.bbox_callback,
            qos_profile=self.isaac_qos_profile
        )

    def start_capture(self) -> None:
        """Enable bbox capturing"""

        self.is_capturing = True
        print("[BboxCapture] Bbox capturing started.")

    def stop_capture(self) -> None:
        """Disable bbox capturing"""

        self.is_capturing = False
        print("[BboxCapture] Bbox capturing stopped.")

    def bbox_callback(self, msg: FrameBboxes) -> None:
        """Callback to handle incoming bbox messages"""

        if not self.is_capturing:
            return

        self.bboxes[self.cur_frame_id] = msg

        self.cur_frame_id += 1


    def save_data_to(self, file_path: str) -> None:
        """Stops capturing and saves the bboxes pickle to the given file path"""

        if not self.bboxes:
            print("[BboxCapture] No bboxes captured to save, skipping save.")
            return
        
        self.is_capturing = False

        with open(file_path, "wb") as f:
            pickle.dump(self.bboxes, f)

        print(f"[BboxCapture] Saved captured bboxes to: {file_path}")

        self.bboxes = {}  # Clear bboxes after saving
        self.cur_frame_id = 0 # Reset frame id after saving
