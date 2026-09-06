"""This file defines a class to capture videos from the isaac sim over ros2"""

# ==================== Imports ====================
import cv2
from cv_bridge import CvBridge

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from .base_capture import BaseCapture


# ==================== Consts ====================
IMAGE_PUBLISHER_TOPIC_NAME = "/isaac_core/image_rgb"


# ==================== The VideoCapture class ====================
class VideoCapture(Node, BaseCapture):
    """A class to capture video from isaac sim over ros2"""

    def __init__(self, topic: str = IMAGE_PUBLISHER_TOPIC_NAME, name: str = "video_capture_node") -> None:
        """Initialize the VideoCapture class"""

        Node.__init__(self, node_name=name)
        BaseCapture.__init__(self)

        self.topic = topic
        self.is_capturing = False
        self.frames = []
        self.bridge = CvBridge()

        self.isaac_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=10
        )

        self.sub = self.create_subscription(
            Image,
            self.topic,
            self.image_callback,
            qos_profile=self.isaac_qos_profile
        )

    def start_capture(self) -> None:
        """Enable video capturing"""

        self.is_capturing = True
        print("[VideoCapture] Video capturing started.")

    def stop_capture(self) -> None:
        """Disable video capturing"""

        self.is_capturing = False
        print("[VideoCapture] Video capturing stopped.")

    def image_callback(self, msg: Image) -> None:
        """Callback to handle incoming image messages"""

        if not self.is_capturing:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.frames.append(cv_image)
        except Exception as e:
            print(f"[VideoCapture] Error converting image message to OpenCV image: {e}")


    def save_data_to(self, file_path: str, fps: int) -> None:
        """Stops capturing and saves the captured video to the given file path"""
        
        if not self.frames:
            print("[VideoCapture] No frames captured to save, skipping save.")
            return
        
        self.is_capturing = False

        height, width, _ = self.frames[0].shape
        video_writer = cv2.VideoWriter(file_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for frame in self.frames:
            video_writer.write(frame)

        video_writer.release()

        print(f"[VideoCapture] Saved captured video to: {file_path}")

        self.frames = []  # Clear frames after saving
