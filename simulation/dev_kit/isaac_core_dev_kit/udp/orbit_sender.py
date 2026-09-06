"""This file defines a class to send an orbit/circle path over UDP to Isaac Sim."""

# ==================== Imports ====================
import math 
from typing import Optional, Tuple

from .base_udp_sender import BaseUDPSender
from .udp_utils import meters_to_latlon_offset


# ==================== the OrbitSender class ====================
class OrbitSender(BaseUDPSender):
    """Sends a circular orbit around a center point over UDP."""

    def __init__(self,
                 center_lat: float, center_lon: float,
                 radius_m: float, height_m: float,
                 speed_mps: float, duration_s: float,
                 roll_deg: float = 0.0, pitch_deg: float = 0.0,
                 udp_port: int = 33333, send_rate_hz: float = 30.0,
                 broadcast: bool = True, target_ip: str = "127.0.0.1"
                 ) -> None:
        """Initialize the OrbitSender class with circle and udp params"""

        super().__init__(udp_port=udp_port, send_rate_hz=send_rate_hz,
                        broadcast=broadcast, target_ip=target_ip)
        
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_m = radius_m
        self.height_m = height_m
        self.speed_mps = speed_mps
        self.duration_s = duration_s
        self.roll = math.radians(roll_deg)
        self.pitch = math.radians(pitch_deg)

        self._total_steps = int(self.duration_s * self.send_rate_hz)

        loop_length = math.pi * 2.0 * self.radius_m
        self._num_cycles = (self.speed_mps * self.duration_s) / loop_length


    def get_next_point(self, step: int) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Returns the next orbit pose."""

        if step >= self._total_steps:
            return None
        
        t = math.pi * 2.0 * self._num_cycles * (step / self._total_steps)

        x = self.radius_m * math.cos(t)
        y = self.radius_m * math.sin(t)

        d_lat, d_lon = meters_to_latlon_offset(x, y, self.center_lat)
        lat = self.center_lat + d_lat
        lon = self.center_lon + d_lon
        alt = self.height_m

        t_next = math.pi * 2.0 * self._num_cycles * ((step + 1) / self._total_steps)
        x_next = self.radius_m * math.cos(t_next)
        y_next = self.radius_m * math.sin(t_next)
        dx = x_next - x
        dy = y_next - y
        yaw = math.atan2(dy, dx)

        return (lat, lon, alt, self.roll, self.pitch, yaw)
 