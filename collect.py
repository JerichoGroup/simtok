"""Orchestrate data collection from Isaac Sim for the SimTok pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from time import sleep, time
from typing import Optional

logger = logging.getLogger(__name__)

import rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Empty

import isaac_core_dev_kit.dev_utils as core_utils
from isaac_core_dev_kit.core_capture.bbox_capture import BboxCapture
from isaac_core_dev_kit.core_capture.pose_capture import PoseCapture
from isaac_core_dev_kit.core_capture.video_capture import VideoCapture
from isaac_core_dev_kit.isaac_manager.host_isaac_manager import HostIsaacManager
from isaac_core_dev_kit.udp.udp_bot import UdpBot
from simtok_utils.zoom_manager import ZoomCommander
from config import get_config


SAMPLE_PATTERN = re.compile(r"sample_(\d{3})_pov_\d+\.mp4")


@dataclass(frozen=True)
class PovConfig:
    """Store a single camera point-of-view configuration."""

    id: int
    forward_m: float = 0.0
    right_m: float = 0.0
    up_m: float = 0.0
    zoom: float = 0.0


class DataCollector:
    """Collect simulation data by recording video, pose, and bbox from Isaac Sim."""

    def __init__(
        self,
        *,
        num_samples: Optional[int] = None,
        video_duration_sec: Optional[int] = None,
        video_fps: Optional[int] = None,
        data_root: Optional[Path] = None,
        usd_path: Optional[str] = None,
        povs: Optional[list[PovConfig]] = None,
    ) -> None:
        """Initialize the collector, prioritizing explicit args over TOML config."""
        config = get_config()
        paths = config.paths
        collect = config.collect
        target = config.collect_target

        self._num_samples: int = num_samples if num_samples is not None else collect["num_samples"]
        self._video_duration_sec: int = video_duration_sec if video_duration_sec is not None else collect["video_duration_sec"]
        self._video_fps: int = video_fps if video_fps is not None else collect["video_fps"]
        self._usd_path: str = usd_path if usd_path is not None else paths["usd_path"]

        self._initial_scene_load_time_sec: int = collect["initial_scene_load_time_sec"]
        self._camera_settle_time_sec: int = collect["camera_settle_time_sec"]

        root = Path(data_root) if data_root is not None else Path(paths["data_root"])
        self._video_dir: Path = root / "videos"
        self._pose_dir: Path = root / "poses"
        self._bbox_dir: Path = root / "bboxes"

        self._target_lat: float = target["lat"]
        self._target_lon: float = target["lon"]
        self._target_alt: float = target["alt"]
        self._default_roll: float = target["roll"]
        self._default_pitch: float = target["pitch"]
        self._default_yaw: float = target["yaw"]

        if povs is not None:
            self._povs = povs
        else:
            self._povs = [
                PovConfig(
                    id=pov["id"],
                    forward_m=pov.get("forward_m", 0.0),
                    right_m=pov.get("right_m", 0.0),
                    up_m=pov.get("up_m", 0.0),
                    zoom=pov.get("zoom", 0.0),
                )
                for pov in config.collect_povs
            ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Execute the full data-collection run."""
        self._create_output_directories()
        core_utils.safe_rclpy_init()

        oscillation_node, reset_pub, new_pub = self._create_oscillation_publishers()

        sim = HostIsaacManager(
            usd_path=self._usd_path,
            com_udp=True,
            bbox_publisher=True,
            sat=True,
            core_path=".",
            show_isaac_logs=False,
        )

        camera = UdpBot(
            self._target_lat,
            self._target_lon,
            self._target_alt,
            self._default_roll,
            self._default_pitch,
            self._default_yaw,
        )

        video = VideoCapture()
        pose = PoseCapture()
        bbox = BboxCapture()
        zoom_commander = ZoomCommander(init_ros=False)

        try:
            with sim:
                sleep(self._initial_scene_load_time_sec)

                video.spin()
                pose.spin()
                bbox.spin()
                sleep(1.0)

                while (
                    reset_pub.get_subscription_count() == 0
                    or new_pub.get_subscription_count() == 0
                ):
                    logger.info("Waiting for thermal node...")
                    sleep(0.1)

                self._generate_dataset(camera, video, pose, bbox, reset_pub, new_pub, zoom_commander)
        finally:
            video.shutdown()
            pose.shutdown()
            bbox.shutdown()
            zoom_commander.close()
            oscillation_node.destroy_node()
            core_utils.safe_rclpy_shutdown()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _create_output_directories(self) -> None:
        """Create the output directories for videos, poses, and bboxes."""
        for directory in (self._video_dir, self._pose_dir, self._bbox_dir):
            directory.mkdir(parents=True, exist_ok=True)

    def _get_next_sample_id(self) -> int:
        """Return the next available sample ID based on existing files."""
        sample_ids = []
        for video in self._video_dir.glob("sample_*_pov_*.mp4"):
            match = SAMPLE_PATTERN.match(video.name)
            if match:
                sample_ids.append(int(match.group(1)))
        return (max(sample_ids) + 1) if sample_ids else 0

    def _build_filename_prefix(self, sample_id: int, pov: PovConfig) -> str:
        """Build the filename prefix for a given sample and POV."""
        return f"sample_{sample_id:03d}_pov_{pov.id}"

    def _move_camera(self, camera: UdpBot, pov: PovConfig, zoom_commander: ZoomCommander) -> None:
        """Move the camera to the given POV and orient it toward the target.

        Start co-located with the target using the default orientation, then
        displace by the POV's relative offsets while keeping that orientation
        fixed, then turn to look back at the target. Relative moves are
        immediate (duration_s=0).

        The reset uses look_at_target=True with turn_duration_s=0 so that
        move_to_point's trailing _turn_to() restores the camera to the default
        orientation (yaw) before the relative moves. Without this the camera
        would retain the heading left by the previous POV's turn_to_point, and
        the yaw-dependent forward/right moves would accumulate rotation error.
        """
        camera.move_to_point(
            self._target_lat, self._target_lon, self._target_alt,
            self._default_roll, self._default_pitch, self._default_yaw,
            look_at_target=True, duration_s=0, turn_duration_s=0
        )

        if pov.forward_m:
            camera.move_forward_backward(pov.forward_m, duration_s=0)
        if pov.right_m:
            camera.move_right_left(pov.right_m, duration_s=0)
        if pov.up_m:
            camera.move_up_down(pov.up_m, duration_s=0)

        camera.turn_to_point(self._target_lat, self._target_lon, self._target_alt)

        zoom_commander.set_zoom(pov.zoom)

        sleep(self._camera_settle_time_sec)

    @staticmethod
    def _publish_oscillation(publisher) -> None:
        """Publish an oscillation control command.

        Delivery is guaranteed by the RELIABLE + TRANSIENT_LOCAL QoS on the
        publisher: the sample is latched and delivered even to a subscriber
        that matches slightly after the publish call.
        """
        publisher.publish(Empty())

    def _start_recording(self, video, pose, bbox) -> None:
        """Start capture on all recording nodes."""
        video.start_capture()
        pose.start_capture()
        bbox.start_capture()

    def _stop_recording(self, video, pose, bbox) -> None:
        """Stop capture on all recording nodes."""
        bbox.stop_capture()
        pose.stop_capture()
        video.stop_capture()

    def _save_recording(self, sample_id: int, pov: PovConfig, video, pose, bbox, fps: float) -> None:
        """Save captured data to disk for a given sample and POV."""
        prefix = self._build_filename_prefix(sample_id, pov)
        video.save_data_to(str(self._video_dir / f"{prefix}.mp4"), fps)
        pose.save_data_to(str(self._pose_dir / f"{prefix}.pkl"))
        bbox.save_data_to(str(self._bbox_dir / f"{prefix}.pkl"))

    def _capture_sample_from_pov(
        self, sample_id, pov, camera, video, pose, bbox, reset_pub, zoom_commander
    ) -> None:
        """Record a single sample from one point of view."""
        logger.info("Recording dataset sample %03d | POV %d", sample_id, pov.id)
        self._move_camera(camera, pov, zoom_commander)
        self._publish_oscillation(reset_pub)
        start_time = time()
        self._start_recording(video, pose, bbox)
        sleep(self._video_duration_sec)
        self._stop_recording(video, pose, bbox)
        elapsed = time() - start_time
        actual_fps = len(video.frames) / elapsed if elapsed > 0 else self._video_fps
        self._save_recording(sample_id, pov, video, pose, bbox, actual_fps)

    def _generate_dataset(self, camera, video, pose, bbox, reset_pub, new_pub, zoom_commander) -> None:
        """Generate all dataset samples across all configured POVs."""
        start_sample = self._get_next_sample_id()
        end_sample = start_sample + self._num_samples

        for sample_id in range(start_sample, end_sample):
            logger.info("=" * 60)
            logger.info("Generating dataset sample %03d", sample_id)
            logger.info("=" * 60)

            self._publish_oscillation(new_pub)

            for pov in self._povs:
                self._capture_sample_from_pov(
                    sample_id, pov, camera, video, pose, bbox, reset_pub, zoom_commander
                )

    @staticmethod
    def _create_oscillation_publishers():
        """Create ROS2 publishers for oscillation reset and new-profile topics."""
        grayscale = get_config().grayscale

        node = rclpy.create_node("oscillation_control_publisher")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        reset_pub = node.create_publisher(Empty, grayscale["reset_oscillation_topic"], qos)
        new_pub = node.create_publisher(Empty, grayscale["new_oscillation_topic"], qos)
        return node, reset_pub, new_pub
