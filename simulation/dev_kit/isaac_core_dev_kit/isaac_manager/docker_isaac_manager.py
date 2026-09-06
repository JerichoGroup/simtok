"""Context manager for launching Isaac Sim inside a Docker container with live log monitoring."""

# ==================== Imports ====================
from types import TracebackType
from pathlib import Path
import subprocess
import threading
import itertools
import tempfile
import time
import yaml
import os


# ==================== Consts ====================
READY_LOG = "rclpy loaded"
DEFAULT_CORE_PATH = "."
DEFAULT_COMPOSE_REL_PATH = "docker/simulation_docker/docker-compose.yml"
CONTAINER_NAME = "isaacsim_2023_ros_humble_core_simulation"
CONTAINER_CORE_PATH = "/root/isaac_core_2023"


# ==================== The DockerIsaacManager class ====================
class DockerIsaacManager:
    """A context manager to start and stop Isaac Sim inside Docker."""

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
                 compose_rel_path: str = DEFAULT_COMPOSE_REL_PATH) -> None:
        """Initialize the context manager with the flags to start Isaac Sim in Docker."""

        self.core_path = Path(core_path).resolve()
        self.compose_template_path = (self.core_path / compose_rel_path).resolve()
        self.show_isaac_logs = show_isaac_logs
        self.container_core_path = Path(CONTAINER_CORE_PATH)

        usd_path = Path(usd_path)
        # make usd_path host absolute
        if not usd_path.is_absolute():
            host_usd_path = (self.core_path / usd_path).resolve()
        else:
            host_usd_path = usd_path

        # make usd_path container absolute
        try:
            relative = host_usd_path.relative_to(self.core_path)
        except ValueError:
            raise RuntimeError("USD path must be inside the core_path directory")

        container_usd_path = self.container_core_path / relative
        usd_path = str(container_usd_path)


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

        self.ready_log = READY_LOG
        self.ready_event = threading.Event()

        self.temp_compose_path: Path | None = None
        self.logs_process: subprocess.Popen | None = None
        self.logs_thread: threading.Thread | None = None

    def _build_core_cmd(self) -> str:
        """Build the Isaac Sim command that will run inside the container."""

        cmd_parts = [
            "source /opt/ros/humble/setup.bash",
            "source /root/IsaacSim-ros_workspaces/humble_ws/install/setup.bash",
        ]

        sim_cmd = ["/isaac-sim/python.sh", "./simulation/main_sim.py"]

        for key, value in self.flags.items():
            if not value:
                continue

            flag = self.FLAG_MAP[key]

            if isinstance(value, str):
                sim_cmd.extend([flag, value])
            else:
                sim_cmd.append(flag)

        sim_cmd_str = " ".join(sim_cmd)

        full_cmd = f"{cmd_parts[0]}; {cmd_parts[1]}; {sim_cmd_str}"

        return full_cmd

    def _create_temp_compose(self) -> None:
        """Create a temporary docker-compose file with a patched command."""

        if not self.compose_template_path.is_file():
            raise FileNotFoundError(f"docker-compose template not found at: {self.compose_template_path}")

        with open(self.compose_template_path, "r") as f:
            compose_data = yaml.safe_load(f)

        full_cmd = self._build_core_cmd()

        try:
            first_service_name = next(iter(compose_data["services"]))
            compose_data["services"][first_service_name]["command"] = [full_cmd]
        except Exception as e:
            raise RuntimeError(
                f"Failed to update 'command' for service {first_service_name} in temp compose file"
                ) from e

        temp_dir = self.compose_template_path.parent
        fd, temp_path_str = tempfile.mkstemp(
            prefix="docker-compose.generated.",
            suffix=".yml",
            dir=temp_dir
        )
        os.close(fd)

        self.temp_compose_path = Path(temp_path_str)

        with open(self.temp_compose_path, 'w') as f:
            yaml.safe_dump(compose_data, f, sort_keys=False)

    def _docker_compose_up(self) -> None:
        """Run 'docker compose up -d' with the temporary compose file."""

        if self.temp_compose_path is None:
            raise RuntimeError("Temporary compose file not created.")

        cmd = [
            "docker", "compose",
            "-f", str(self.temp_compose_path),
            "up", "-d"
        ]

        subprocess.check_call(cmd, cwd=self.core_path)

    def _docker_compose_down(self) -> None:
        """Run 'docker compose down' with the temporary compose file."""

        if self.temp_compose_path is None or not self.temp_compose_path.exists():
            return

        cmd = [
            "docker", "compose",
            "-f", str(self.temp_compose_path),
            "down"
        ]

        try:
            subprocess.check_call(cmd, cwd=self.core_path)
        except Exception as e:
            print(f"Exception caught while running docker compose down: {e}")

    def _stream_logs(self) -> None:
        """Stream docker logs from the container and detect readiness."""

        cmd = ["docker", "logs", "-f", CONTAINER_NAME]

        self.logs_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        try:
            for line in iter(self.logs_process.stdout.readline, ""):
                if not line:
                    break

                line = line.rstrip()
                if self.show_isaac_logs:
                    print(f"[DOCKER ISAAC SIM] {line}")

                if self.ready_log in line:
                    self.ready_event.set()
        finally:
            if self.logs_process and self.logs_process.poll() is None:
                try:
                    self.logs_process.terminate()
                except Exception:
                    pass

    def _wait_with_msg(self, timeout: float = 300, buffer: float = 0.25) -> None:
        """Animated wait until the ready flag appears in logs."""

        start_time = time.time()
        symbols = itertools.cycle(["-", "\\", "|", "/"])

        while not self.ready_event.is_set():
            if time.time() - start_time > timeout:
                print(f"timeout reached while waiting for: {self.ready_log}")
                return

            print(f"waiting for IsaacSim (docker) to load...  {next(symbols)}", end="\r")
            time.sleep(buffer)

        time.sleep(5)
        print(f"\rIsaacSim (docker) loaded in {time.time() - start_time:.2f} seconds! {' ' * 30}")

    def _cleanup(self) -> None:
        """Stop docker compose and remove the temporary compose file."""

        self._docker_compose_down()

        if self.temp_compose_path and self.temp_compose_path.exists():
            try:
                self.temp_compose_path.unlink()
            except Exception as e:
                print(f"Exception caught while trying to delete temp file: {e}")

    def __enter__(self) -> "DockerIsaacManager":
        """Start Isaac Sim in Docker and wait for it to finish loading."""

        try:
            print("Starting IsaacSim in Docker...")

            self.ready_event.clear()

            self._create_temp_compose()

            self._docker_compose_up()

            self.logs_thread = threading.Thread(
                target=self._stream_logs,
                daemon=True
            )
            self.logs_thread.start()

            self._wait_with_msg()

            return self

        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected during Docker startup, cleaning up...")
            self._cleanup()
            raise
        except Exception as e:
            print(f"\nAn exception occurred during Docker IsaacSim startup: {e}")
            self._cleanup()
            raise

    def __exit__(self,
                 exc_type: type[BaseException] | None,
                 exc_value: BaseException | None,
                 exc_traceback: TracebackType | None) -> bool | None:
        """Stop Isaac Sim Docker stack and clean up temporary files."""

        if exc_type:
            print(f"an exception occurred while in DockerIsaacManager context: {exc_value}")

        print("closing IsaacSim (docker)...")
        try:
            self._cleanup()
        except KeyboardInterrupt:
            print("\nKeyboardInterrupt detected during Docker shutdown, forcing cleanup...")
            self._cleanup()
        print("IsaacSim (docker) closed.")
