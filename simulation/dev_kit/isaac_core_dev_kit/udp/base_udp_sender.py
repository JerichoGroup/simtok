"""Base classes for sending UDP control packets to control Isaac Sim."""

# ==================== Imports ====================
import time
import socket
import struct
import threading
from typing import Tuple, Optional
from abc import ABC, abstractmethod


# ==================== The BaseUDPSender class ====================
class BaseUDPSender(ABC):
    """Base class for UDP senders that control Isaac Sim objects."""

    PACKET_FORMAT = "<6d"
    HEADER1 = 0xAC
    HEADER2 = 0xDC

    def __init__(
        self,
        udp_port: int = 33333,
        send_rate_hz: float = 30.0,
        broadcast: bool = True,
        target_ip: str = "127.0.0.1",
    ) -> None:
        """Initialize the BaseUDPSender with the target IP address."""

        self.udp_port = udp_port
        self.send_rate_hz = send_rate_hz
        self.broadcast = broadcast
        self.target_ip = target_ip

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._running = False
        self._thread: Optional[threading.Thread] = None

        if self.broadcast:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)            

    @abstractmethod
    def get_next_point(self, step: int) -> Optional[Tuple[float, float, float, float, float, float]]:
        """Return the next point as: (lat, lon, alt, roll, pitch, yaw) or None if the sending should stop."""
        pass

    def _build_packet(self, lat: float, lon: float, alt: float,
                      roll: float, pitch: float, yaw: float) -> bytes:
        """Build a UDP packet with the expected Isaac Sim format."""

        fields = [lat, lon, alt, roll, pitch, yaw]
        payload = struct.pack(self.PACKET_FORMAT, *fields)

        checksum = 0
        for byte in payload:
            checksum ^= byte

        packet = bytes([self.HEADER1, self.HEADER2]) + payload + bytes([checksum])
        return packet

    def send_once(self, lat: float, lon: float, alt: float,
                  roll: float, pitch: float, yaw: float) -> None:
        """Send a single packet with the given pose."""

        packet = self._build_packet(lat, lon, alt, roll, pitch, yaw)
        self._sock.sendto(packet, (self.target_ip, self.udp_port))

    def _run_loop(self, max_steps: Optional[int] = None) -> None:
        """Run the sender loop until stopped or max_steps is reached."""

        self._running = True
        cur_step = 0
        dt = 1.0 / self.send_rate_hz
        next_time = time.perf_counter()

        try:
            while self._running:
                if max_steps is not None and cur_step >= max_steps:
                    break

                point = self.get_next_point(cur_step)
                if point is None:
                    break

                lat, lon, alt, roll, pitch, yaw = point
                packet = self._build_packet(lat, lon, alt, roll, pitch, yaw)
                self._sock.sendto(packet, (self.target_ip, self.udp_port))

                cur_step += 1
                next_time += dt
                sleep_time = next_time - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"Error in UDP sender to {self.udp_port}: {e}")
        finally:
            self._running = False
            self._thread = None

    def run(self, max_steps: Optional[int] = None, blocking: bool = False) -> None:
        """Runs the sender loop in a blocking or non-blocking way"""

        if blocking:
            self._run_loop(max_steps)
            return
        
        if self._thread is not None and self._thread.is_alive():
            print("[BaseUDPSender] sender already running")
            return
        
        self._thread = threading.Thread(target=self._run_loop,
                                        args=(max_steps,),
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the sender loop."""

        self._running = False

    def join(self, timeout: Optional[float] = 0.1) -> None:
        """Wait for the sender thread to finish"""

        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def close(self) -> None:
        """Close the underlying socket."""

        try:
            self._sock.close()
        except Exception:
            pass
