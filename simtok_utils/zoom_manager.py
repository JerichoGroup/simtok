from __future__ import annotations

import threading
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


DEFAULT_ZOOM_TOPIC = "/simtok/zoom"


class ZoomPublisher:
    """Publishes zoom values to a ROS2 topic."""

    def __init__(
        self,
        node: Node,
        topic: str = DEFAULT_ZOOM_TOPIC,
    ) -> None:

        self._publisher = node.create_publisher(
            Float32,
            topic,
            10,
        )

    def publish(self, zoom: float) -> None:
        msg = Float32()
        msg.data = zoom
        self._publisher.publish(msg)


class ZoomController:
    """Maintains the current zoom and publishes updates."""

    def __init__(
        self,
        publisher: ZoomPublisher,
    ) -> None:

        self._publisher = publisher
        self._zoom = 0.0
        self._lock = threading.Lock()

    @property
    def zoom(self) -> float:
        with self._lock:
            return self._zoom

    def set_zoom(self, zoom: float) -> float:
        """Immediately set the zoom."""

        zoom = max(0.0, min(1.0, float(zoom)))

        self._publisher.publish(zoom)

        with self._lock:
            self._zoom = zoom

        return zoom


class ZoomSlewer:
    """Smoothly slews a ZoomController to a target value."""

    def __init__(
        self,
        controller: ZoomController,
    ) -> None:

        self._controller = controller
        self._generation = 0
        self._lock = threading.Lock()

    def slew_to(
        self,
        target: float,
        duration_sec: float = 4.0,
        hz: float = 20.0,
    ) -> threading.Thread:

        target = max(0.0, min(1.0, float(target)))

        with self._lock:
            self._generation += 1
            generation = self._generation

        start = self._controller.zoom

        def run() -> None:

            steps = max(1, int(duration_sec * hz))

            for step in range(1, steps + 1):

                with self._lock:
                    if generation != self._generation:
                        return

                alpha = step / steps

                zoom = start + (target - start) * alpha

                self._controller.set_zoom(zoom)

                time.sleep(1.0 / hz)

        thread = threading.Thread(
            target=run,
            daemon=True,
        )
        thread.start()

        return thread

    def cancel(self) -> None:
        with self._lock:
            self._generation += 1


class ZoomCommander:
    """
    High-level interface for controlling camera zoom.

    Combines a publisher, controller and slewer.
    """

    def __init__(
        self,
        topic: str = DEFAULT_ZOOM_TOPIC,
        node_name: str = "zoom_commander",
        init_ros: bool = True,
    ) -> None:

        if init_ros and not rclpy.ok():
            rclpy.init()

        self.node = rclpy.create_node(node_name)

        publisher = ZoomPublisher(
            self.node,
            topic,
        )

        self._controller = ZoomController(
            publisher,
        )

        self._slewer = ZoomSlewer(
            self._controller,
        )

    @property
    def zoom(self) -> float:
        return self._controller.zoom

    def set_zoom(self, zoom: float) -> float:
        return self._controller.set_zoom(zoom)

    def slew_to(
        self,
        target: float,
        duration_sec: float = 4.0,
        hz: float = 20.0,
    ) -> threading.Thread:
        return self._slewer.slew_to(
            target,
            duration_sec,
            hz,
        )

    def close(self) -> None:

        self._slewer.cancel()

        try:
            self.node.destroy_node()
        except Exception:
            pass