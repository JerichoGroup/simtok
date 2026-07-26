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

    # Extract bbox target data
    # Flip Z sign: pkl stores (target - camera), we want positive z = camera is above object
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
        description="Combine pose and bbox pkl files into a per-frame JSON output."
    )

    parser.add_argument(
        "--pose-pkl",
        type=Path,
        required=True,
        help="Path to the pose pickle file (GeoPoseStamped messages).",
    )

    parser.add_argument(
        "--bbox-pkl",
        type=Path,
        required=True,
        help="Path to the bbox pickle file (FrameBboxes messages).",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("data/jsons"),
        help="Output path. If a directory, the JSON filename is auto-derived from the matching video in data/videos/.",
    )

    return parser.parse_args()


def find_matching_video_name(pkl_path: Path) -> str:
    """
    Find the matching video name from data/videos/ based on the pkl file's numeric suffix.
    
    e.g. test_bbox_1.pkl -> looks for a video containing '_1' like test_video_pov_1.mp4
    Falls back to the pkl stem if no matching video is found.
    """
    import re

    # Extract trailing number from pkl stem (e.g. "test_bbox_1" -> "1")
    match = re.search(r"(\d+)$", pkl_path.stem)
    if not match:
        return pkl_path.stem

    suffix_num = match.group(1)
    videos_dir = Path("data/videos")

    if videos_dir.is_dir():
        # Look for a video file that ends with _<num>.mp4 (not _<num>_something.mp4)
        pattern = re.compile(rf"^(.+)_{re.escape(suffix_num)}\.(?:mp4|avi|mkv|mov)$")
        for video_file in sorted(videos_dir.iterdir()):
            if video_file.is_file() and pattern.match(video_file.name):
                return video_file.stem

    # Fallback to pkl stem
    return pkl_path.stem


def resolve_output_path(args) -> Path:
    """
    If -o is a directory, auto-name the JSON after the matching video.
    If -o is a file path, use it directly.
    """
    output = args.output

    if output.is_dir() or not output.suffix:
        # Treat as directory — auto-derive filename from matching video
        output.mkdir(parents=True, exist_ok=True)
        video_name = find_matching_video_name(args.bbox_pkl)
        return output / f"{video_name}.json"

    return output


def main():
    args = parse_args()

    # Load pickle files
    pose_data = load_pkl(args.pose_pkl)
    bbox_data = load_pkl(args.bbox_pkl)

    # Build per-frame JSON
    result = build_json(pose_data, bbox_data)

    # Write output
    output_path = resolve_output_path(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(result, f, indent=4)

    print(f"[json_creation] Wrote {result['total_frames']} frames to {output_path}")


if __name__ == "__main__":
    main()
