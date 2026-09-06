"""Unit tests for the pure math in the UDP bot module.

These tests cover only deterministic input->output math (no sockets, no ROS,
no timing, no Isaac Sim):
    - UdpBot._smooth_rotation_matrixs_interpolation: quaternion SLERP
    - UdpBot._normalize_angle_r: angle wrapping to [-pi, pi)
    - udp_utils.lla_distance_to_meters: haversine great-circle distance
    - udp_utils.meters_to_latlon_offset: local flat-earth meters->degrees

The math is checked against known/analytic values.

UdpBot.__init__ opens a UDP socket (via BaseUDPSender), so tests that need an
instance build one with __new__ to bypass __init__ — the methods under test
(SLERP, angle normalization) do not touch instance state.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Make the udp package importable without a conftest.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_UDP_PKG_ROOT = _PROJECT_ROOT / "simulation" / "dev_kit" / "isaac_core_dev_kit"
if str(_UDP_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_UDP_PKG_ROOT))

from transforms3d.euler import euler2mat, mat2euler

from udp.udp_bot import UdpBot, EULER_AXES
from udp.udp_utils import (
    R,
    LLAPoint,
    lla_distance_to_meters,
    meters_to_latlon_offset,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _bare_bot() -> UdpBot:
    """Build a UdpBot without running __init__ (avoids opening a socket)."""
    return UdpBot.__new__(UdpBot)


def _yaw_deg_of(rotation_matrix: np.ndarray) -> float:
    """Return the yaw component (degrees) of a rotation matrix."""
    _, _, yaw_r = mat2euler(rotation_matrix, axes=EULER_AXES)
    return math.degrees(yaw_r)


# ----------------------------------------------------------------------
# SLERP — endpoints, midpoint, and shortest-path behaviour
# ----------------------------------------------------------------------

def test_slerp_endpoint_t0_returns_start():
    """At progress 0, SLERP returns the start orientation."""
    bot = _bare_bot()
    start = np.eye(3)
    end = euler2mat(0.0, 0.0, math.pi / 2, axes=EULER_AXES)
    result = bot._smooth_rotation_matrixs_interpolation(start, end, 0.0)
    assert np.allclose(result, start, atol=1e-9)


def test_slerp_endpoint_t1_returns_end():
    """At progress 1, SLERP returns the end orientation."""
    bot = _bare_bot()
    start = np.eye(3)
    end = euler2mat(0.0, 0.0, math.pi / 2, axes=EULER_AXES)
    result = bot._smooth_rotation_matrixs_interpolation(start, end, 1.0)
    assert np.allclose(result, end, atol=1e-9)


def test_slerp_midpoint_is_half_angle():
    """Halfway through a 90-degree yaw turn is exactly 45 degrees."""
    bot = _bare_bot()
    start = np.eye(3)
    end = euler2mat(0.0, 0.0, math.pi / 2, axes=EULER_AXES)
    result = bot._smooth_rotation_matrixs_interpolation(start, end, 0.5)
    assert _yaw_deg_of(result) == pytest.approx(45.0, abs=1e-9)


@pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_slerp_constant_angular_velocity(alpha):
    """SLERP interpolates the angle linearly: yaw at alpha == alpha * total."""
    bot = _bare_bot()
    total_deg = 80.0
    start = np.eye(3)
    end = euler2mat(0.0, 0.0, math.radians(total_deg), axes=EULER_AXES)
    result = bot._smooth_rotation_matrixs_interpolation(start, end, alpha)
    assert _yaw_deg_of(result) == pytest.approx(alpha * total_deg, abs=1e-6)


def test_slerp_result_is_valid_rotation_matrix():
    """SLERP output must be a proper rotation matrix (orthonormal, det=+1)."""
    bot = _bare_bot()
    start = euler2mat(math.radians(10), math.radians(20), math.radians(30), axes=EULER_AXES)
    end = euler2mat(math.radians(-40), math.radians(15), math.radians(70), axes=EULER_AXES)
    result = bot._smooth_rotation_matrixs_interpolation(start, end, 0.37)
    assert np.allclose(result @ result.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(result) == pytest.approx(1.0, abs=1e-9)


def test_slerp_takes_shortest_path():
    """A 270-degree yaw target is reached the short way (-90 degrees)."""
    bot = _bare_bot()
    start = np.eye(3)
    # 270 deg CCW == -90 deg; shortest path midpoint should be -45 deg, not +135.
    end = euler2mat(0.0, 0.0, math.radians(270), axes=EULER_AXES)
    result = bot._smooth_rotation_matrixs_interpolation(start, end, 0.5)
    assert _yaw_deg_of(result) == pytest.approx(-45.0, abs=1e-6)


def test_slerp_close_quaternions_use_lerp_branch():
    """A tiny rotation exercises the near-parallel (lerp) branch and stays valid."""
    bot = _bare_bot()
    start = np.eye(3)
    end = euler2mat(0.0, 0.0, math.radians(0.1), axes=EULER_AXES)  # dot > DOT_THRESHOLD
    result = bot._smooth_rotation_matrixs_interpolation(start, end, 0.5)
    assert _yaw_deg_of(result) == pytest.approx(0.05, abs=1e-4)
    assert np.allclose(result @ result.T, np.eye(3), atol=1e-9)


# ----------------------------------------------------------------------
# _normalize_angle_r — wrap to (-pi, pi]
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "angle_r, expected_r",
    [
        (0.0, 0.0),
        (math.pi / 2, math.pi / 2),
        (-math.pi / 2, -math.pi / 2),
        # The formula is (a + pi) % (2pi) - pi, i.e. the range is [-pi, pi):
        # +pi and its odd multiples fold onto -pi.
        (math.pi, -math.pi),
        (-math.pi, -math.pi),
        (3 * math.pi, -math.pi),
        (-3 * math.pi, -math.pi),
        (2 * math.pi, 0.0),           # full turn -> 0
        (math.pi + 0.1, -math.pi + 0.1),  # just past +pi wraps negative
    ],
)
def test_normalize_angle_known_values(angle_r, expected_r):
    """Angle normalization maps into [-pi, pi) with known results."""
    assert UdpBot._normalize_angle_r(angle_r) == pytest.approx(expected_r, abs=1e-12)


@pytest.mark.parametrize("k", [-5, -2, -1, 0, 1, 2, 5])
def test_normalize_angle_is_2pi_periodic(k):
    """Adding whole turns of 2pi does not change the normalized angle."""
    base = 0.7  # arbitrary angle already in range
    result = UdpBot._normalize_angle_r(base + k * 2.0 * math.pi)
    assert result == pytest.approx(base, abs=1e-9)


def test_normalize_angle_always_in_range():
    """Output is always within [-pi, pi) for a wide sweep of inputs."""
    for deg in range(-1080, 1081, 7):
        out = UdpBot._normalize_angle_r(math.radians(deg))
        assert -math.pi - 1e-12 <= out < math.pi + 1e-12


# ----------------------------------------------------------------------
# lla_distance_to_meters — haversine great-circle distance
# ----------------------------------------------------------------------

def test_distance_zero_for_same_point():
    """Distance between a point and itself is exactly zero."""
    p = LLAPoint(32.0, 34.0, 100.0)
    assert lla_distance_to_meters(p, p) == 0.0


def test_distance_one_degree_latitude():
    """One degree of latitude equals R * 1 degree in radians (meridian arc)."""
    d = lla_distance_to_meters(LLAPoint(0.0, 0.0, 0.0), LLAPoint(1.0, 0.0, 0.0))
    assert d == pytest.approx(R * math.radians(1.0), rel=1e-12)


def test_distance_one_degree_longitude_at_equator():
    """At the equator, one degree of longitude also equals R * 1 degree."""
    d = lla_distance_to_meters(LLAPoint(0.0, 0.0, 0.0), LLAPoint(0.0, 1.0, 0.0))
    assert d == pytest.approx(R * math.radians(1.0), rel=1e-9)


def test_distance_longitude_shrinks_with_latitude():
    """One degree of longitude at 60 deg latitude is ~half of that at the equator.

    The cos(lat) scaling is exact only in the small-angle limit; across a full
    1-degree span the haversine result differs by ~1e-5 relative, so the
    tolerance is set accordingly.
    """
    d_equator = lla_distance_to_meters(LLAPoint(0.0, 0.0, 0.0), LLAPoint(0.0, 1.0, 0.0))
    d_60 = lla_distance_to_meters(LLAPoint(60.0, 0.0, 0.0), LLAPoint(60.0, 1.0, 0.0))
    assert d_60 == pytest.approx(d_equator * math.cos(math.radians(60.0)), rel=1e-4)


def test_distance_is_symmetric():
    """Distance is the same regardless of argument order."""
    a = LLAPoint(10.0, 20.0, 0.0)
    b = LLAPoint(11.5, 21.25, 0.0)
    assert lla_distance_to_meters(a, b) == pytest.approx(lla_distance_to_meters(b, a), rel=1e-12)


def test_distance_ignores_altitude():
    """The haversine formula uses only lat/lon; altitude does not affect it."""
    a = LLAPoint(10.0, 20.0, 0.0)
    b = LLAPoint(10.0, 20.0, 5000.0)
    assert lla_distance_to_meters(a, b) == 0.0


# ----------------------------------------------------------------------
# meters_to_latlon_offset — local flat-earth conversion
# ----------------------------------------------------------------------

def test_offset_northing_maps_to_latitude():
    """A northing of R*1deg yields exactly +1 degree latitude, 0 longitude."""
    d_lat, d_lon = meters_to_latlon_offset(R * math.radians(1.0), 0.0, ref_lat=0.0)
    assert d_lat == pytest.approx(1.0, rel=1e-12)
    assert d_lon == 0.0


def test_offset_easting_at_equator():
    """At the equator, an easting of R*1deg yields +1 degree longitude."""
    d_lat, d_lon = meters_to_latlon_offset(0.0, R * math.radians(1.0), ref_lat=0.0)
    assert d_lat == 0.0
    assert d_lon == pytest.approx(1.0, rel=1e-12)


def test_offset_longitude_scales_with_latitude():
    """Longitude degrees per meter grow by 1/cos(lat) away from the equator."""
    _, d_lon_eq = meters_to_latlon_offset(0.0, 1000.0, ref_lat=0.0)
    _, d_lon_60 = meters_to_latlon_offset(0.0, 1000.0, ref_lat=60.0)
    # 1/cos(60deg) == 2, so the same easting spans twice as many degrees.
    assert d_lon_60 == pytest.approx(d_lon_eq / math.cos(math.radians(60.0)), rel=1e-9)
    assert d_lon_60 == pytest.approx(2.0 * d_lon_eq, rel=1e-9)


def test_offset_latitude_independent_of_ref_lat():
    """Latitude offset depends only on northing, not on the reference latitude."""
    d_lat_0, _ = meters_to_latlon_offset(1234.0, 0.0, ref_lat=0.0)
    d_lat_45, _ = meters_to_latlon_offset(1234.0, 0.0, ref_lat=45.0)
    assert d_lat_0 == pytest.approx(d_lat_45, rel=1e-12)


def test_offset_sign_convention():
    """Negative northing/easting produce negative latitude/longitude offsets."""
    d_lat, d_lon = meters_to_latlon_offset(-500.0, -500.0, ref_lat=30.0)
    assert d_lat < 0.0
    assert d_lon < 0.0


# ----------------------------------------------------------------------
# Round-trip consistency between the two LLA helpers
# ----------------------------------------------------------------------

def test_meridian_roundtrip_distance_then_offset():
    """Going N by a known distance then measuring it returns the same distance."""
    ref_lat = 0.0
    northing_m = R * math.radians(0.5)  # half a degree along the meridian
    d_lat, _ = meters_to_latlon_offset(northing_m, 0.0, ref_lat)
    measured = lla_distance_to_meters(
        LLAPoint(ref_lat, 0.0, 0.0),
        LLAPoint(ref_lat + d_lat, 0.0, 0.0),
    )
    assert measured == pytest.approx(northing_m, rel=1e-9)
