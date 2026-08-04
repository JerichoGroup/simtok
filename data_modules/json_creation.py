#!/usr/bin/env python3
"""
Combines a pose pkl and a bbox pkl into a per-frame JSON file.

Pose pkl structure: dict[frame_id (int) -> GeoPoseStamped msg]
    GeoPoseStamped.pose.orientation stores RPY as: x=roll, y=pitch, z=yaw

Bbox pkl structure: dict[frame_id (int) -> FrameBboxes msg]
    FrameBboxes.bboxes is a list of Bbox msgs, each with:
        target_name, in_frame, is_visible, distance_x, distance_y, distance_z
"""

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, List


def load_pkl(path: Path) -> Dict[int, Any]:
    """Load a pickle file and return its contents."""
    with path.open("rb") as f:
        return pickle.load(f)


def extract_frame_data(pose_msg: Any, bbox_msg: Any) -> Dict[str, Any]:
    """
    Extract relevant fields from a single frame's pose and bbox messages.

    Args:
        pose_msg: A GeoPoseStamped message (orientation.y = pitch, orientation.z = yaw)
        bbox_msg: A FrameBboxes message (bboxes list with distance and target info)

    Returns:
        Dictionary with per-frame data including pitch, yaw, and bbox target info.
    """
    # Extract orientation from pose (stored in radians, ENU frame)
    # GeoPoseStamped: msg.pose.orientation where x=roll_enu, y=pitch_enu, z=yaw_enu
    # Convert back to NED aircraft convention for output:
    #   NED pitch = ENU roll (orientation.x)
    #   NED yaw = -(yaw_enu - pi/2)
    orientation = pose_msg.pose.orientation
    pitch = math.degrees(float(orientation.x))
    yaw_enu = float(orientation.z)
    yaw = math.degrees(-(yaw_enu - math.pi / 2))
    # Normalise yaw to [-180, 180]
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

    frame_entry = {
        "pitch": pitch,
        "yaw": yaw,
        "targets": targets,
    }

    return frame_entry


def build_json(pose_data: Dict[int, Any], bbox_data: Dict[int, Any]) -> Dict[str, Any]:
    """
    Build the full JSON structure by iterating over all frames.

    Aligns frames by frame_id (integer keys present in both pkls).
    """
    # Use frames that exist in both pkl files
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Combine all pose/bbox PKL pairs into per-frame JSON files."
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Dataset root directory.",
    )

    return parser.parse_args()


def find_matching_bbox_file(
    pose_file: Path,
    bbox_directory: Path,
) -> Path | None:
    """Return the matching bbox PKL if it exists."""

    bbox_file = bbox_directory / pose_file.name

    if not bbox_file.exists():
        return None

    return bbox_file


def find_pose_files(pose_directory: Path) -> List[Path]:
    """Return all pose PKL files."""

    pose_files = sorted(pose_directory.glob("*.pkl"))

    if not pose_files:
        raise FileNotFoundError(
            f"No pose PKLs found in '{pose_directory}'."
        )

    return pose_files


def write_json_file(
    output_path: Path,
    json_data: Dict[str, Any],
) -> None:
    """Write JSON data to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(json_data, f, indent=4)


def process_pkl_pair(
    pose_file: Path,
    bbox_file: Path,
    output_directory: Path,
) -> None:
    """Convert one pose/bbox pair into a JSON file."""

    pose_data = load_pkl(pose_file)
    bbox_data = load_pkl(bbox_file)

    result = build_json(pose_data, bbox_data)

    output_path = output_directory / f"{pose_file.stem}.json"

    write_json_file(output_path, result)

    print(
        f"[json_creation] "
        f"Wrote {result['total_frames']} frames to {output_path}"
    )


def process_dataset(data_root: Path) -> None:
    """Generate JSON files for the entire dataset."""

    pose_directory = data_root / "poses"
    bbox_directory = data_root / "bboxes"
    json_directory = data_root / "jsons"

    pose_files = find_pose_files(pose_directory)

    for pose_file in pose_files:

        bbox_file = find_matching_bbox_file(
            pose_file,
            bbox_directory,
        )

        if bbox_file is None:
            print(f"[WARNING] Missing bbox file: {pose_file.name}")
            continue

        process_pkl_pair(
            pose_file,
            bbox_file,
            json_directory,
        )


def find_matching_bbox_file(
    pose_file: Path,
    bbox_directory: Path,
) -> Path | None:
    """Return the matching bbox PKL if it exists."""

    bbox_file = bbox_directory / pose_file.name

    if not bbox_file.exists():
        return None

    return bbox_file


def main() -> None:
    args = parse_args()
    process_dataset(args.data_root)


if __name__ == "__main__":
    main()
