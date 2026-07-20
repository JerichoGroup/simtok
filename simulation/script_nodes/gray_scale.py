"""this file implements a script node that subscribe to a ros2 topic and capture an image to a specified path"""

# ==================== imports ====================
import rclpy
import threading
from rclpy.node import Node
from isaac_ros2_messages.msg import FrameBboxes
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy


# ==================== consts ====================
BBOX_TOPIC_NAME = "/isaac_core/bbox"
CUBE_X = 0.0
CUBE_Y = 0.0
CUBE_Z = 3.0
TARGET_NAME = "Cube"
PRIM_PATH = "/bboxes/Cube"


# ==================== the ROS2BboxNode class ====================
class ROS2BboxNode:
    """this class subscribes to a topic for output paths"""

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
        self.current_rgb = None

    def bbox_callback(self, msg: FrameBboxes) -> None:
        """callback for BBOXOutput messages"""

        self.current_distance = #TODO calc distance from msg
        self.current_rgb = #TODO extract RGB from distance

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
    """Capture image if a new output path is received via ROS2"""

    target_rgb = db.internal_state.ros2_bbox_node.current_rgb

    #TODO: write the rgb values to the prim at PRIM_PATH

    return True

# ==================== cleanup
def cleanup(db):
    """Reset internal state"""

    try:
        db.internal_state.ros2_bbox_node.node.destroy_node()
    except Exception as e:
        pass
    try:
        rclpy.shutdown()
    except Exception as e:
        pass

    db.internal_state.ros2_bbox_node = None
