"""Generate per-frame JSON metadata from pose and bbox pickle files."""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import get_config


def load_pkl(path: Path) -> Dict[int, Any]:
    """Load and return the contents of a pickle file."""
    with path.open("rb") as f:
        return pickle.load(f)


def extract_frame_data(pose_msg: Any, bbox_msg: Any) -> Dict[str, Any]:
    """Extract pitch, yaw, and target distances from a single frame's messages."""
    orientation = pose_msg.pose.orientation
    pitch = math.degrees(float(orientation.x))
    yaw_enu = float(orientation.z)
    yaw = math.degrees(-(yaw_enu - math.pi / 2))
    yaw = (yaw + 180) % 360 - 180

    targets: List[Dict[str, Any]] = []
    for bbox in bbox_msg.bboxes:
        dx = float(bbox.distance_x)
        dy = float(bbox.distance_y)
        dz = -float(bbox.distance_z)
        distance_xy = math.sqrt(dx**2 + dy**2)
        distance_xyz = math.sqrt(dx**2 + dy**2 + dz**2)

        targets.append({
            "target_name": str(bbox.target_name),
            "in_frame": bool(bbox.in_frame),
            "is_visible": bool(bbox.is_visible),
            "distance_x": dx,
            "distance_y": dy,
            "distance_z": dz,
            "distance_xy": round(distance_xy, 4),
            "distance_xyz": round(distance_xyz, 4),
        })

    return {
        "pitch": pitch,
        "yaw": yaw,
        "targets": targets,
    }


def build_json(pose_data: Dict[int, Any], bbox_data: Dict[int, Any]) -> Dict[str, Any]:
    """Build the full JSON structure from aligned pose and bbox frame data."""
    common_frames = sorted(set(pose_data.keys()) & set(bbox_data.keys()))

    if not common_frames:
        raise ValueError(
            f"No common frame IDs between pose pkl (frames: {sorted(pose_data.keys())[:5]}...) "
            f"and bbox pkl (frames: {sorted(bbox_data.keys())[:5]}...)"
        )

    frames: Dict[str, Any] = {}
    for frame_id in common_frames:
        frame_entry = extract_frame_data(pose_data[frame_id], bbox_data[frame_id])
        frames[str(frame_id)] = frame_entry

    return {
        "total_frames": len(frames),
        "frames": frames,
    }


def find_matching_bbox_file(pose_file: Path, bbox_directory: Path) -> Path | None:
    """Return the matching bbox PKL path if it exists, otherwise None."""
    bbox_file = bbox_directory / pose_file.name
    if not bbox_file.exists():
        return None
    return bbox_file


def find_pose_files(pose_directory: Path) -> List[Path]:
    """Return all pose PKL files sorted from the given directory."""
    pose_files = sorted(pose_directory.glob("*.pkl"))
    if not pose_files:
        raise FileNotFoundError(
            f"No pose PKLs found in '{pose_directory}'."
        )
    return pose_files


def write_json_file(output_path: Path, json_data: Dict[str, Any]) -> None:
    """Write a JSON dict to disk, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(json_data, f, indent=4)


def process_pkl_pair(pose_file: Path, bbox_file: Path, output_directory: Path) -> None:
    """Convert one pose/bbox PKL pair into a JSON file."""
    pose_data = load_pkl(pose_file)
    bbox_data = load_pkl(bbox_file)

    result = build_json(pose_data, bbox_data)

    output_path = output_directory / f"{pose_file.stem}.json"
    write_json_file(output_path, result)

    print(
        f"[json_creation] "
        f"Wrote {result['total_frames']} frames to {output_path}"
    )


class JsonCreator:
    """Generate per-frame JSON metadata for the entire dataset."""

    def __init__(self, data_root: Optional[Path] = None) -> None:
        """Initialize the creator, prioritizing explicit data_root over TOML config."""
        cfg = get_config()
        paths = cfg.paths

        resolved_root = data_root if data_root is not None else Path(paths["data_root"])

        self._pose_dir: Path = resolved_root / "poses"
        self._bbox_dir: Path = resolved_root / "bboxes"
        self._json_dir: Path = resolved_root / "jsons"

    def process_dataset(self) -> None:
        """Generate JSON files for all matching pose/bbox pairs in the dataset."""
        pose_files = find_pose_files(self._pose_dir)

        for pose_file in pose_files:
            bbox_file = find_matching_bbox_file(pose_file, self._bbox_dir)

            if bbox_file is None:
                print(f"[WARNING] Missing bbox file: {pose_file.name}")
                continue

            process_pkl_pair(pose_file, bbox_file, self._json_dir)
