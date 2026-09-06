"""Context manager for launching Isaac Sim on your host machine with live stdout monitoring."""

# ==================== Imports ====================
from types import TracebackType
from pathlib import Path
import subprocess
import threading
import itertools
import signal
import time
import os


# ==================== Consts ====================
READY_LOG = "rclpy loaded"
DEFAULT_CORE_PATH = '.'
DEFAULT_ISAAC_PATH = Path.home() / ".local" / "share" / "ov" / "pkg" / "isaac_sim-2023.1.1"


# ==================== The HostIsaacManager class ====================
class HostIsaacManager:
    """A context manager to start and stop isaac sim"""

    FLAGS = [
        "usd_path",
        "headless",
        "com_ros",
        "com_udp",
        "distance_sensor",
        "bbox_publisher",
        "sat",
        "image_rtp",
    ]

    FLAG_MAP = {flag: f"--{flag.replace('_', '-')}" for flag in FLAGS}

    def __init__(self,
                 usd_path: str = "usd/maps/earth/earth.usda",
                 headless: bool = False,
                 com_ros: bool = False,
                 com_udp: bool = False,
                 distance_sensor: bool = False,
                 bbox_publisher: bool = False,
                 sat: bool = False,
                 image_rtp: bool = False,
                 show_isaac_logs: bool = False,
                 core_path: str = DEFAULT_CORE_PATH,
                 isaac_path: str = DEFAULT_ISAAC_PATH) -> None:
        """Initialize the context manager with the flags to start isaac sim on your host machine"""

        self.core_path = Path(core_path).resolve()
        self.isaac_path = Path(isaac_path).resolve()
        self.show_isaac_logs = show_isaac_logs

        if not Path(usd_path).is_absolute():
            usd_path = str(self.core_path / usd_path)
        usd_path = str(usd_path)

        self.flags = {
            "usd_path": usd_path,
            "headless": headless,
            "com_ros": com_ros,
            "com_udp": com_udp,
            "distance_sensor": distance_sensor,
            "bbox_publisher": bbox_publisher,
            "sat": sat,
            "image_rtp": image_rtp,
        }

        self.process = None
        self.stdout_thread = None
        self.ready_log = READY_LOG
        self.ready_event = threading.Event()

        self.isaac_python_abs_path = self.isaac_path / "python.sh"
        self.main_sim_path = self.core_path / "simulation" / "main_sim.py"
        self.isaac_core_cmd = []

    def _build_isaac_core_cmd(self) -> None:
        """Build the isaac core command with the given flags"""

        self.isaac_core_cmd = [
            str(self.isaac_python_abs_path),
            str(self.main_sim_path)
        ]

        for key, value in self.flags.items():
            if not value:
                continue

            flag = self.FLAG_MAP[key]

            if isinstance(value, str):
                self.isaac_core_cmd.extend([flag, value])
            elif isinstance(value, bool) and value:
                self.isaac_core_cmd.append(flag)

    def _stream_stdout(self) -> None:
        """Read stdout line-by-line and detect readiness"""

        for line in iter(self.process.stdout.readline, ""):
            if not line:
                break

            line = line.rstrip()
            if self.show_isaac_logs:
                print(f"[HOST ISAAC SIM] {line}")

            if self.ready_log in line:
                self.ready_event.set()

    def _wait_with_msg(self, timeout: float = 300, buffer: float = 0.25) -> None:
        """Animated wait until the ready flag appears in stdout"""

        start_time = time.time()
        symbols = itertools.cycle(["-", "\\", "|", "/"])

        while not self.ready_event.is_set():
            if time.time() - start_time > timeout:
                print(f"timeout reached while waiting for: {self.ready_log}")
                return

            print(f"waiting for IsaacSim to load...  {next(symbols)}", end="\r")
            time.sleep(buffer)
        
        time.sleep(5)

        print(f"\rIsaacSim loaded in {time.time() - start_time:.2f} seconds! {' ' * 30}")

    def _cleanup_process_group(self) -> None:
        """Kill the entire process group of the isaac sim process"""

        if not self.process:
            return

        try:
            pgid = os.getpgid(self.process.pid)
            os.killpg(pgid, signal.SIGTERM)
            time.sleep(1)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def __enter__(self) -> "HostIsaacManager":
        """Starts isaac sim on your host machine and wait for it to finish loading"""

        try:
            print("Starting IsaacSim...")
            self._build_isaac_core_cmd()

            self.ready_event.clear()

            cmd_str = " ".join(self.isaac_core_cmd)
            print(f"current isaac cmd:\n\t{cmd_str}")

            self.process = subprocess.Popen(
                ["bash", "-c", cmd_str],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid    # Unite all child processes into a single process group
            )

            self.stdout_thread = threading.Thread(
                target=self._stream_stdout,
                daemon=True
            )
            self.stdout_thread.start()

            self._wait_with_msg()

            return self

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected during startup, Cleaning up IsaacSim...")
            self._cleanup_process_group()
            raise
        except Exception as e:
            print(f"\nAn exception occurred during IsaacSim startup: {e}")
            self._cleanup_process_group()
            raise


    def __exit__(self,
                 exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 exc_traceback: TracebackType | None) -> bool | None:
        """Kill all isaac sim processes after exiting the context"""

        if exc_type:
            print(f"an exception occurred while in HostIsaacManager context: {exc_value}")

        print("closing IsaacSim...")
        self._cleanup_process_group()
        print("IsaacSim closed.")
