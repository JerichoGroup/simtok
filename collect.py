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
    depth_m: float = 0.0
    horizontal_m: float = 0.0
    vertical_m: float = 0.0
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
        use_random_povs: Optional[bool] = None,
        povs: Optional[list[PovConfig]] = None,
    ) -> None:
        """Initialize the collector, prioritizing explicit args over TOML config."""
        config = get_config()

        self._resolve_run_params(config, num_samples, video_duration_sec, video_fps, usd_path)
        self._build_output_dirs(config, data_root)
        self._unpack_target(config.collect_target)
        self._resolve_povs(config, use_random_povs, povs)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _resolve_run_params(
        self,
        config,
        num_samples: Optional[int],
        video_duration_sec: Optional[int],
        video_fps: Optional[int],
        usd_path: Optional[str],
    ) -> None:
        """Resolve scalar run parameters, preferring explicit args over TOML."""
        collect = config.collect

        self._num_samples: int = num_samples if num_samples is not None else collect["num_samples"]
        self._video_duration_sec: int = video_duration_sec if video_duration_sec is not None else collect["video_duration_sec"]
        self._video_fps: int = video_fps if video_fps is not None else collect["video_fps"]
        self._usd_path: str = usd_path if usd_path is not None else config.paths["usd_path"]

        self._initial_scene_load_time_sec: int = collect["initial_scene_load_time_sec"]
        self._camera_settle_time_sec: int = collect["camera_settle_time_sec"]

    def _build_output_dirs(self, config, data_root: Optional[Path]) -> None:
        """Derive the video/pose/bbox output directories from the data root."""
        root = Path(data_root) if data_root is not None else Path(config.paths["data_root"])
        self._video_dir: Path = root / "videos"
        self._pose_dir: Path = root / "poses"
        self._bbox_dir: Path = root / "bboxes"

    def _unpack_target(self, target: dict) -> None:
        """Unpack the target location and default orientation."""
        self._target_lat: float = target["lat"]
        self._target_lon: float = target["lon"]
        self._target_alt: float = target["alt"]
        self._default_roll: float = target["roll"]
        self._default_pitch: float = target["pitch"]
        self._default_yaw: float = target["yaw"]

    def _resolve_povs(
        self,
        config,
        use_random_povs: Optional[bool],
        povs: Optional[list[PovConfig]],
    ) -> None:
        """Determine how POVs are produced for the run and store the mode.

        Sets:
          - self._use_random: whether POVs are regenerated randomly per sample.
          - self._random_cfg: the [collect.random] config used for regeneration.
          - self._povs: the fixed POV list for the explicit/configured paths
            (unused when self._use_random is True, where POVs are drawn fresh
            for each sample).

        Precedence: an explicit ``povs`` argument wins, then the random flag
        (arg over TOML ``enabled``), else the configured [[collect.povs]] list.
        In random mode the configured POVs are ignored entirely.
        """
        random_cfg = config.collect_random
        self._random_cfg = random_cfg

        if povs is not None:
            self._use_random = False
            self._povs = povs
            return

        use_random = (
            use_random_povs
            if use_random_povs is not None
            else random_cfg.get("enabled", False)
        )
        self._use_random = bool(use_random)

        if self._use_random:
            self._povs = []
        else:
            self._povs = self._load_configured_povs(config.collect_povs)

    def _povs_for_sample(self) -> list[PovConfig]:
        """Return the POVs to use for one sample.

        In random mode a fresh list of random POVs is generated for every
        sample (ids 1..num_povs). Otherwise the fixed configured/explicit
        list is reused across all samples.
        """
        if self._use_random:
            return self._generate_random_povs(self._random_cfg)
        return self._povs

    @staticmethod
    def _load_configured_povs(pov_entries: list[dict]) -> list[PovConfig]:
        """Build a list of POVs from the [[collect.povs]] config entries.

        The config expresses depth as a positive "behind the target" distance;
        it is negated into the internal depth_m so that depth_m < 0 places the
        camera behind the target (the convention used elsewhere).
        """
        return [
            PovConfig(
                id=pov["id"],
                depth_m=-pov.get("depth_m", 0.0),
                horizontal_m=pov.get("horizontal_m", 0.0),
                vertical_m=pov.get("vertical_m", 0.0),
                zoom=pov.get("zoom", 0.0),
            )
            for pov in pov_entries
        ]

    @staticmethod
    def _generate_random_povs(random_cfg: dict) -> list[PovConfig]:
        """Build a list of random POVs from the [collect.random] config."""
        # Imported lazily to avoid a circular import (random_povs imports PovConfig).
        from random_povs import RandomPovGenerator

        generator = RandomPovGenerator(
            x_range=tuple(random_cfg["depth_range_m"]),
            y_range=tuple(random_cfg["horizontal_range_m"]),
            z_range=tuple(random_cfg["vertical_range_m"]),
            zoom_range=tuple(random_cfg.get("zoom_range", (0.0, 0.0))),
        )
        return generator.generate(random_cfg["num_povs"])

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

        if pov.depth_m:
            camera.move_forward_backward(pov.depth_m, duration_s=0)
        if pov.horizontal_m:
            camera.move_right_left(pov.horizontal_m, duration_s=0)
        if pov.vertical_m:
            camera.move_up_down(pov.vertical_m, duration_s=0)

        camera.turn_to_point(self._target_lat, self._target_lon, self._target_alt, 0)

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

            for pov in self._povs_for_sample():
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
