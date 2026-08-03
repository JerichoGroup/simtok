from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import sleep
import rclpy
from std_msgs.msg import Empty
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import isaac_core_dev_kit.dev_utils as core_utils
from isaac_core_dev_kit.core_capture.bbox_capture import BboxCapture
from isaac_core_dev_kit.core_capture.pose_capture import PoseCapture
from isaac_core_dev_kit.core_capture.video_capture import VideoCapture
from isaac_core_dev_kit.isaac_manager.host_isaac_manager import HostIsaacManager
from isaac_core_dev_kit.udp.udp_bot import UdpBot

# =============================================================================
# Configuration
# =============================================================================

USD_PATH = "/home/user/clones/simtok/usd/maps/scenes/cube.usda"

DATA_ROOT = Path("./data")
VIDEO_DIR = DATA_ROOT / "videos"
POSE_DIR = DATA_ROOT / "poses"
BBOX_DIR = DATA_ROOT / "bboxes"

RESET_OSCILLATION_TOPIC = "/isaac_core/reset_oscillation"

NUM_SAMPLES = 10
VIDEO_DURATION_SEC = 60
VIDEO_FPS = 36

INITIAL_SCENE_LOAD_TIME_SEC = 5
CAMERA_SETTLE_TIME_SEC = 2
VERTICAL_MOVE_SETTLE_TIME_SEC = 0.5

TARGET_LAT = 32.12345
TARGET_LON = 35.12345
TARGET_ALT = 0.0

DEFAULT_ROLL = 0.0
DEFAULT_PITCH = 0.0
DEFAULT_YAW = 0.0


@dataclass(frozen=True)
class PovConfig:
    id: int
    lat: float
    lon: float
    alt: float
    move_up_m: float = 0.0
    pitch_deg: float = 0.0


CAMERA_POVS = (
    PovConfig(1, 32.123269, 35.1232421, 0.0, move_up_m=5.0, pitch_deg=-5.0),
    PovConfig(2, 32.1234500, 35.1233561, 0.0),
)


def create_output_directories():
    for dir in (VIDEO_DIR, POSE_DIR, BBOX_DIR):
        dir.mkdir(parents=True, exist_ok=True)


def build_sample_filename_prefix(sample_id: int, pov: PovConfig) -> str:
    return f"sample_{sample_id:03d}_pov_{pov.id}"


def move_camera(camera: UdpBot, pov: PovConfig):
    camera.move_to_point(
        pov.lat, pov.lon, pov.alt,
        DEFAULT_ROLL, DEFAULT_PITCH, DEFAULT_YAW,
        look_at_target=False,
    )

    camera.turn_to_point(TARGET_LAT, TARGET_LON, TARGET_ALT)

    if pov.move_up_m:
        camera.move_up_down(pov.move_up_m)
        sleep(VERTICAL_MOVE_SETTLE_TIME_SEC)

    if pov.pitch_deg:
        camera.turn_pitch(pov.pitch_deg)

    sleep(CAMERA_SETTLE_TIME_SEC)


def start_recording(video_capture: VideoCapture, pose_capture: PoseCapture, bbox_capture: BboxCapture):
    video_capture.start_capture()
    pose_capture.start_capture()
    bbox_capture.start_capture()


def stop_recording(video_capture: VideoCapture, pose_capture: PoseCapture, bbox_capture: BboxCapture):
    bbox_capture.stop_capture()
    pose_capture.stop_capture()
    video_capture.stop_capture()


def save_recording(sample_id: int, pov: PovConfig,
                   video_capture: VideoCapture, pose_capture: PoseCapture, bbox_capture: BboxCapture):

    prefix = build_sample_filename_prefix(sample_id, pov)

    video_capture.save_data_to(str(VIDEO_DIR / f"{prefix}.mp4"), VIDEO_FPS)
    pose_capture.save_data_to(str(POSE_DIR / f"{prefix}.pkl"))
    bbox_capture.save_data_to(str(BBOX_DIR / f"{prefix}.pkl"))


def capture_sample_from_pov(sample_id: int, pov: PovConfig,
                   camera: UdpBot,
                   video_capture: VideoCapture,
                   pose_capture: PoseCapture,
                   bbox_capture: BboxCapture):

    print(f"Recording sample {sample_id + 1}/{NUM_SAMPLES} | POV {pov.id}")

    move_camera(camera, pov)
    start_recording(video_capture, pose_capture, bbox_capture)
    sleep(VIDEO_DURATION_SEC)
    stop_recording(video_capture, pose_capture, bbox_capture)
    save_recording(sample_id, pov, video_capture, pose_capture, bbox_capture)


def reset_oscillation(reset_publisher: Publisher) -> None:
    msg = Empty()

    for _ in range(2):
        reset_publisher.publish(msg)
        sleep(0.05)
    sleep(0.2)


def create_reset_publisher():
    node = rclpy.create_node("oscillation_reset_publisher")

    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    publisher = node.create_publisher(
        Empty,
        RESET_OSCILLATION_TOPIC,
        qos,
    )

    return node, publisher


def generate_dataset(
    camera: UdpBot,
    video_capture: VideoCapture,
    pose_capture: PoseCapture,
    bbox_capture: BboxCapture,
    reset_publisher,
) -> None:

    for sample_id in range(NUM_SAMPLES):
        print("=" * 60)
        print(f"Generating sample {sample_id + 1}/{NUM_SAMPLES}")
        print("=" * 60)

        reset_oscillation(reset_publisher)
        
        for pov in CAMERA_POVS:
            capture_sample_from_pov(
                sample_id,
                pov,
                camera,
                video_capture,
                pose_capture,
                bbox_capture,
            )

def main():

    create_output_directories()
    core_utils.safe_rclpy_init()

    reset_node, reset_publisher = create_reset_publisher()

    sim = HostIsaacManager(
        usd_path=USD_PATH,
        com_udp=True,
        bbox_publisher=True,
        sat=True,
        core_path=".",
        show_isaac_logs=False,
    )

    camera = UdpBot(
        TARGET_LAT,
        TARGET_LON,
        TARGET_ALT,
        DEFAULT_ROLL,
        DEFAULT_PITCH,
        DEFAULT_YAW,
    )

    video = VideoCapture()
    pose = PoseCapture()
    bbox = BboxCapture()

    with sim:
        sleep(INITIAL_SCENE_LOAD_TIME_SEC)
        video.spin()
        pose.spin()
        bbox.spin()

        sleep(1.0)

        while reset_publisher.get_subscription_count() == 0:
            print("Waiting for oscillation reset subscriber...")
            sleep(0.1)


        generate_dataset(
            camera,
            video,
            pose,
            bbox,
            reset_publisher,
        )

    video.shutdown()
    pose.shutdown()
    bbox.shutdown()

    reset_node.destroy_node()
    core_utils.safe_rclpy_shutdown()


if __name__ == "__main__":
    main()