"""Unit tests for the pure JSON-conversion logic in data_modules.json_creation.

These tests cover only pure input->output logic (no disk I/O, no ROS, no config):
    - extract_frame_data: pitch/yaw conversion and per-target distance math
    - build_json: frame-ID intersection, structure, and error handling

Messages are faked with SimpleNamespace since the functions rely only on
attribute access (pose_msg.pose.orientation.y, bbox_msg.bboxes[].distance_x).
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make the project root importable without a conftest.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_modules.json_creation import build_json, extract_frame_data


# ----------------------------------------------------------------------
# Fake message builders
# ----------------------------------------------------------------------

def make_pose(orientation_x=0.0, orientation_y=0.0, orientation_z=0.0):
    """Build a fake pose message.

    NOTE: orientation here carries Euler angles (radians), not a quaternion —
    the upstream publisher stuffs roll/pitch/yaw into x/y/z. extract_frame_data
    reads .y as pitch and .z as yaw (ENU).
    """
    orientation = SimpleNamespace(x=orientation_x, y=orientation_y, z=orientation_z, w=1.0)
    pose = SimpleNamespace(orientation=orientation)
    return SimpleNamespace(pose=pose)


def make_bbox(
    target_name="Cube",
    in_frame=True,
    is_visible=True,
    distance_x=0.0,
    distance_y=0.0,
    distance_z=0.0,
):
    """Build a single fake bbox entry."""
    return SimpleNamespace(
        target_name=target_name,
        in_frame=in_frame,
        is_visible=is_visible,
        distance_x=distance_x,
        distance_y=distance_y,
        distance_z=distance_z,
    )


def make_bbox_msg(*bboxes):
    """Build a fake bbox message wrapping zero or more bbox entries."""
    return SimpleNamespace(bboxes=list(bboxes))


# ----------------------------------------------------------------------
# extract_frame_data — yaw (ENU -> compass heading, then radians -> degrees)
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "yaw_enu_rad, expected_yaw_deg",
    [
        (math.pi / 2, 0.0),      # ENU East-CCW pi/2 == pointing North  -> compass 0
        (0.0, 90.0),             # pointing East                        -> compass 90
        (math.pi, -90.0),        # pointing West                        -> compass -90
        (-math.pi / 2, -180.0),  # pointing South                       -> compass +/-180
    ],
)
def test_yaw_enu_to_compass(yaw_enu_rad, expected_yaw_deg):
    """Yaw should be re-referenced from ENU (East/CCW) to compass (North/CW)."""
    result = extract_frame_data(make_pose(orientation_z=yaw_enu_rad), make_bbox_msg())
    assert result["yaw"] == pytest.approx(expected_yaw_deg, abs=1e-9)


def test_yaw_is_wrapped_into_180_range():
    """Yaw must always land within (-180, 180]."""
    for yaw_enu in [-10.0, -math.pi, 0.0, math.pi, 10.0, 100.0]:
        yaw = extract_frame_data(make_pose(orientation_z=yaw_enu), make_bbox_msg())["yaw"]
        assert -180.0 < yaw <= 180.0


# ----------------------------------------------------------------------
# extract_frame_data — pitch (radians -> degrees only)
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "pitch_rad, expected_pitch_deg",
    [
        (0.0, 0.0),
        (math.pi / 6, 30.0),
        (-math.pi / 4, -45.0),
        (math.pi, 180.0),
    ],
)
def test_pitch_radians_to_degrees(pitch_rad, expected_pitch_deg):
    """Pitch (read from orientation.y) is a plain radians->degrees conversion."""
    result = extract_frame_data(make_pose(orientation_y=pitch_rad), make_bbox_msg())
    assert result["pitch"] == pytest.approx(expected_pitch_deg, abs=1e-9)


# ----------------------------------------------------------------------
# extract_frame_data — target distances
# ----------------------------------------------------------------------

def test_distance_z_is_sign_flipped():
    """distance_z is stored negated relative to the raw bbox value."""
    bbox = make_bbox(distance_z=12.0)
    target = extract_frame_data(make_pose(), make_bbox_msg(bbox))["targets"][0]
    assert target["distance_z"] == -12.0


def test_distance_xy_and_xyz_math():
    """distance_xy/xyz use Pythagorean magnitude, rounded to 4 dp."""
    # (3,4) -> 5 ;  (3,4,12) -> 13   (dz negated to -12 but squared, so magnitude same)
    bbox = make_bbox(distance_x=3.0, distance_y=4.0, distance_z=12.0)
    target = extract_frame_data(make_pose(), make_bbox_msg(bbox))["targets"][0]
    assert target["distance_x"] == 3.0
    assert target["distance_y"] == 4.0
    assert target["distance_z"] == -12.0
    assert target["distance_xy"] == pytest.approx(5.0)
    assert target["distance_xyz"] == pytest.approx(13.0)


def test_distances_are_rounded_to_4dp():
    """distance_xy/xyz are rounded to 4 decimal places."""
    bbox = make_bbox(distance_x=1.0, distance_y=1.0, distance_z=1.0)
    target = extract_frame_data(make_pose(), make_bbox_msg(bbox))["targets"][0]
    # sqrt(2) = 1.41421356..., sqrt(3) = 1.7320508...
    assert target["distance_xy"] == 1.4142
    assert target["distance_xyz"] == 1.7321


def test_field_types_are_coerced():
    """Values are coerced to native types (str/bool/float) regardless of input type."""
    bbox = make_bbox(
        target_name=123,          # non-str
        in_frame=1,               # truthy int
        is_visible=0,             # falsy int
        distance_x="2.0",         # numeric string
        distance_y=0,
        distance_z=0,
    )
    target = extract_frame_data(make_pose(), make_bbox_msg(bbox))["targets"][0]
    assert target["target_name"] == "123" and isinstance(target["target_name"], str)
    assert target["in_frame"] is True
    assert target["is_visible"] is False
    assert isinstance(target["distance_x"], float) and target["distance_x"] == 2.0


def test_multiple_targets_preserved_in_order():
    """All bboxes are extracted, preserving order."""
    msg = make_bbox_msg(
        make_bbox(target_name="A"),
        make_bbox(target_name="B"),
        make_bbox(target_name="C"),
    )
    targets = extract_frame_data(make_pose(), msg)["targets"]
    assert [t["target_name"] for t in targets] == ["A", "B", "C"]


def test_no_targets_yields_empty_list():
    """A frame with no bboxes yields an empty targets list, not an error."""
    result = extract_frame_data(make_pose(), make_bbox_msg())
    assert result["targets"] == []
    assert set(result.keys()) == {"pitch", "yaw", "targets"}


# ----------------------------------------------------------------------
# build_json — frame intersection, structure, errors
# ----------------------------------------------------------------------

def test_build_json_uses_frame_intersection():
    """Only frame IDs present in BOTH pose and bbox data are included."""
    pose_data = {0: make_pose(), 1: make_pose(), 2: make_pose()}
    bbox_data = {1: make_bbox_msg(), 2: make_bbox_msg(), 3: make_bbox_msg()}

    result = build_json(pose_data, bbox_data)

    assert result["total_frames"] == 2
    assert set(result["frames"].keys()) == {"1", "2"}  # keys stringified; 0 and 3 excluded


def test_build_json_frame_keys_are_strings():
    """Frame IDs become string keys in the output."""
    pose_data = {5: make_pose()}
    bbox_data = {5: make_bbox_msg(make_bbox())}
    result = build_json(pose_data, bbox_data)
    assert "5" in result["frames"]
    assert result["frames"]["5"]["targets"][0]["target_name"] == "Cube"


def test_build_json_structure():
    """Top-level structure has total_frames and a frames dict."""
    pose_data = {0: make_pose(orientation_y=math.pi / 6, orientation_z=0.0)}
    bbox_data = {0: make_bbox_msg(make_bbox(distance_x=3.0, distance_y=4.0, distance_z=12.0))}

    result = build_json(pose_data, bbox_data)

    assert set(result.keys()) == {"total_frames", "frames"}
    assert result["total_frames"] == 1
    frame = result["frames"]["0"]
    assert frame["pitch"] == pytest.approx(30.0)
    assert frame["yaw"] == pytest.approx(90.0)
    assert frame["targets"][0]["distance_xyz"] == pytest.approx(13.0)


def test_build_json_raises_on_no_common_frames():
    """No overlapping frame IDs must raise ValueError, never write bad data silently."""
    pose_data = {0: make_pose(), 1: make_pose()}
    bbox_data = {2: make_bbox_msg(), 3: make_bbox_msg()}

    with pytest.raises(ValueError, match="No common frame IDs"):
        build_json(pose_data, bbox_data)


def test_build_json_raises_on_empty_inputs():
    """Empty pose/bbox data also has no common frames -> ValueError."""
    with pytest.raises(ValueError, match="No common frame IDs"):
        build_json({}, {})
