"""This file Defines a base class for capture modules in Isaac Core."""

# ==================== Imports ====================
import threading
from abc import ABC, abstractmethod

from rclpy.executors import MultiThreadedExecutor

from isaac_core_dev_kit.dev_utils import safe_rclpy_init, safe_rclpy_shutdown


# ==================== The BaseCapture class ====================
class BaseCapture(ABC):
    """Abstract base class for capture modules in Isaac Core."""

    def __init__(self) -> None:
        """Initialize the BaseCapture class."""

        safe_rclpy_init()
        self._executor = MultiThreadedExecutor()
        self._spin_thread = None

    @abstractmethod
    def start_capture(self) -> None:
        """Start the capture process."""
        pass

    @abstractmethod
    def stop_capture(self) -> None:
        """Stop the capture process."""
        pass

    @abstractmethod
    def save_data_to(self, file_path: str) -> None:
        """Saves the captured data to the specified file path."""
        pass

    def spin(self) -> None:
        """Spin capture node in a background thread."""

        self._executor.add_node(self)

        if self._spin_thread is None:
            self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
            self._spin_thread.start()

    def shutdown(self, shutdown_rclpy=False) -> None:
        """Shutdown executor and capture node."""
        
        try:
            if self._executor is not None:
                self._executor.remove_node(self)
                self._executor.shutdown()
            
            if self._spin_thread is not None:
                self._spin_thread.join(timeout=0.1)
            
            self.destroy_node()
            if shutdown_rclpy:
                safe_rclpy_shutdown()
            
            self._executor = None
            self._spin_thread = None
        except Exception as e:
            print(f"[BaseCapture] Error during shutdown: {e}")
