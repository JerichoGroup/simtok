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


USD_PATH = "/home/user/clones/simtok/usd/maps/scenes/cube.usda"

DATA_ROOT = Path("./data")
VIDEO_DIR = DATA_ROOT / "videos"
POSE_DIR = DATA_ROOT / "poses"
BBOX_DIR = DATA_ROOT / "bboxes"


RESET_OSCILLATION_TOPIC = "/simtok/reset_oscillation"
NEW_OSCILLATION_TOPIC = "/simtok/new_oscillation"


NUM_SAMPLES = 10
VIDEO_DURATION_SEC = 60
VIDEO_FPS = 20


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
    PovConfig(
        1,
        32.123269,
        35.1232421,
        0.0,
        move_up_m=5.0,
        pitch_deg=-5.0,
    ),
    PovConfig(
        2,
        32.1234500,
        35.1233561,
        0.0,
    ),
)


def create_output_directories():
    for directory in (VIDEO_DIR, POSE_DIR, BBOX_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def build_sample_filename_prefix(sample_id: int, pov: PovConfig) -> str:
    return f"sample_{sample_id:03d}_pov_{pov.id}"


def move_camera(camera: UdpBot, pov: PovConfig):

    camera.move_to_point(
        pov.lat,
        pov.lon,
        pov.alt,
        DEFAULT_ROLL,
        DEFAULT_PITCH,
        DEFAULT_YAW,
        look_at_target=False,
    )

    camera.turn_to_point(
        TARGET_LAT,
        TARGET_LON,
        TARGET_ALT,
    )

    if pov.move_up_m:
        camera.move_up_down(pov.move_up_m)
        sleep(VERTICAL_MOVE_SETTLE_TIME_SEC)

    if pov.pitch_deg:
        camera.turn_pitch(pov.pitch_deg)

    sleep(CAMERA_SETTLE_TIME_SEC)


def start_recording(
    video_capture: VideoCapture,
    pose_capture: PoseCapture,
    bbox_capture: BboxCapture,
):
    video_capture.start_capture()
    pose_capture.start_capture()
    bbox_capture.start_capture()


def stop_recording(
    video_capture: VideoCapture,
    pose_capture: PoseCapture,
    bbox_capture: BboxCapture,
):
    bbox_capture.stop_capture()
    pose_capture.stop_capture()
    video_capture.stop_capture()


def save_recording(
    sample_id: int,
    pov: PovConfig,
    video_capture: VideoCapture,
    pose_capture: PoseCapture,
    bbox_capture: BboxCapture,
):

    prefix = build_sample_filename_prefix(
        sample_id,
        pov,
    )

    video_capture.save_data_to(
        str(VIDEO_DIR / f"{prefix}.mp4"),
        VIDEO_FPS,
    )

    pose_capture.save_data_to(
        str(POSE_DIR / f"{prefix}.pkl")
    )

    bbox_capture.save_data_to(
        str(BBOX_DIR / f"{prefix}.pkl")
    )


def capture_sample_from_pov(
    sample_id: int,
    pov: PovConfig,
    camera: UdpBot,
    video_capture: VideoCapture,
    pose_capture: PoseCapture,
    bbox_capture: BboxCapture,
    reset_publisher,
):

    print(
        f"Recording sample {sample_id + 1}/{NUM_SAMPLES} | POV {pov.id}"
    )

    move_camera(
        camera,
        pov,
    )

    publish_oscillation_command(
        reset_publisher
    )

    start_recording(
        video_capture,
        pose_capture,
        bbox_capture,
    )

    sleep(VIDEO_DURATION_SEC)

    stop_recording(
        video_capture,
        pose_capture,
        bbox_capture,
    )

    save_recording(
        sample_id,
        pov,
        video_capture,
        pose_capture,
        bbox_capture,
    )


def create_oscillation_publishers():

    node = rclpy.create_node(
        "oscillation_control_publisher"
    )

    qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )

    reset_publisher = node.create_publisher(
        Empty,
        RESET_OSCILLATION_TOPIC,
        qos,
    )

    new_publisher = node.create_publisher(
        Empty,
        NEW_OSCILLATION_TOPIC,
        qos,
    )

    return (
        node,
        reset_publisher,
        new_publisher,
    )


def publish_oscillation_command(
    publisher,
):

    msg = Empty()

    # Publish twice to avoid missing first command
    for _ in range(2):
        publisher.publish(msg)
        sleep(0.05)

    sleep(0.2)


def generate_dataset(
    camera: UdpBot,
    video_capture: VideoCapture,
    pose_capture: PoseCapture,
    bbox_capture: BboxCapture,
    reset_publisher,
    new_publisher,
):

    for sample_id in range(NUM_SAMPLES):

        print("=" * 60)
        print(
            f"Generating sample {sample_id + 1}/{NUM_SAMPLES}"
        )
        print("=" * 60)


        # Generate one thermal profile for this sample
        publish_oscillation_command(
            new_publisher
        )


        # Record every POV using the same thermal profile
        for pov in CAMERA_POVS:

            capture_sample_from_pov(
                sample_id,
                pov,
                camera,
                video_capture,
                pose_capture,
                bbox_capture,
                reset_publisher,
            )


def main():

    create_output_directories()

    core_utils.safe_rclpy_init()


    (
        oscillation_node,
        reset_publisher,
        new_publisher,
    ) = create_oscillation_publishers()


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


        while (
            reset_publisher.get_subscription_count() == 0
            or new_publisher.get_subscription_count() == 0
        ):
            print(
                "Waiting for thermal node..."
            )
            sleep(0.1)


        generate_dataset(
            camera,
            video,
            pose,
            bbox,
            reset_publisher,
            new_publisher,
        )


    video.shutdown()
    pose.shutdown()
    bbox.shutdown()


    oscillation_node.destroy_node()

    core_utils.safe_rclpy_shutdown()


if __name__ == "__main__":
    main()