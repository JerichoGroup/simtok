"""This file defines a class to send a constant lla point over UDP to Isaac Sim."""

# ==================== Imports ====================
from typing import Optional, Tuple

from .base_udp_sender import BaseUDPSender


# ==================== the OnePointSender class ====================
class OnePointSender(BaseUDPSender):
    """Sends a single constant lla point at a fixed rate over UDP."""

    def __init__(self,
                 lat: float, lon: float, alt: float,
                 roll: float, pitch: float, yaw: float,
                 udp_port: float = 33333, send_rate_hz: float = 30,
                 broadcast: bool = True, target_ip: str = "127.0.0.1"
                ) -> None:
        """Initialize the OnePointSender with a point and an address"""

        super().__init__(udp_port=udp_port, send_rate_hz=send_rate_hz,
                         broadcast=broadcast, target_ip=target_ip)
        self._lat = lat
        self._lon = lon
        self._alt = alt
        self._roll = roll
        self._pitch = pitch
        self._yaw = yaw

    def get_next_point(self, step: int = 0) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Always return the same pose."""

        return (self._lat, self._lon, self._alt, self._roll, self._pitch, self._yaw)
