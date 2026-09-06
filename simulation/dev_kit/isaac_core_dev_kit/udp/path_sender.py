"""This file defines a class to send a path over UDP to Isaac Sim."""

# ==================== Imports ====================
import math
from typing import List, Optional, Tuple

from .base_udp_sender import BaseUDPSender
from .udp_utils import LLAPoint, lla_distance_to_meters


# ==================== the PathSender class ====================
class PathSender(BaseUDPSender):
    """Sends a path defined by LLA points at a given speed over UDP."""

    def __init__(self,
                 points: List[LLAPoint],
                 speed_mps: float,
                 roll_deg: float = 0.0, pitch_deg: float = 0.0,
                 udp_port: int = 33333, send_rate_hz: float = 30.0,
                 broadcast: bool = True, target_ip: str = "127.0.0.1"
                 ) -> None:
        """Initialize the PathSender with the path points and udp target params"""

        super().__init__(udp_port=udp_port, send_rate_hz=send_rate_hz,
                        broadcast=broadcast, target_ip=target_ip)
        
        if len(points) < 2:
            raise ValueError("[PathSender] requires at least two LLAPoint objects to run.")
        
        self.points = points
        self.speed_mps = speed_mps
        self.roll = math.radians(roll_deg)
        self.pitch = math.radians(pitch_deg)

        self._distances = [0.0]
        for i in range(1, len(self.points)):
            d = lla_distance_to_meters(points[i - 1], points[i])
            self._distances.append(self._distances[-1] + d)

        self._total_distance = self._distances[-1]
        self._total_duration = self._total_distance / self.speed_mps
        self._total_steps = int(self._total_duration * self.send_rate_hz)


    def get_next_point(self, step: int) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Returns the interpolated pose for the given step."""

        if step >= self._total_steps:
            return None
        
        target_dist = step * self.speed_mps / self.send_rate_hz

        i = 1
        while i < len(self._distances) and self._distances[i] < target_dist:
            i += 1

        if i >= len(self._distances):
            i = len(self._distances) - 1

        d_start = self._distances[i - 1]
        d_end = self._distances[i]
        t = (target_dist - d_start) / (d_end - d_start) if d_end != d_start else 0.0

        p1 = self.points[i - 1]
        p2 = self.points[i]

        lat = p1.lat + t * (p2.lat - p1.lat)
        lon = p1.lon + t * (p2.lon - p1.lon)
        alt = p1.alt + t * (p2.alt - p1.alt)

        d_lat = p2.lat - p1.lat
        d_lon = p2.lon - p1.lon

        yaw = 0.0
        if d_lat != 0.0 or d_lon != 0.0:
            yaw = math.atan2(d_lon, d_lat)

        return (lat, lon, alt, self.roll, self.pitch, yaw)
