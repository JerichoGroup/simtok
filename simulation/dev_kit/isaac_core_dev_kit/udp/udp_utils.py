"""Utility functions for UDP path senders."""

# ==================== Imports ====================
import math
from typing import Tuple
from dataclasses import dataclass


# ==================== Consts ====================
R = 6371000.0


# ==================== Data Classes ====================
@dataclass
class LLAPoint:
    lat: float
    lon: float
    alt: float


# ==================== Helper - lla to distance m ====================
def lla_distance_to_meters(p1: LLAPoint, p2: LLAPoint) -> float:
    """Compute distance in meters between two LLA points."""


    lat1, lon1 = math.radians(p1.lat), math.radians(p1.lon)
    lat2, lon2 = math.radians(p2.lat), math.radians(p2.lon)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2

    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ==================== Helper - meters ====================
def meters_to_latlon_offset(dx: float, dy: float, ref_lat: float) -> Tuple[float, float]:
    """Convert x/y offset in meters to lat/lon offset in degrees."""

    d_lat = dx / R
    d_lon = dy / (R * math.cos(math.radians(ref_lat)))

    return math.degrees(d_lat), math.degrees(d_lon)
