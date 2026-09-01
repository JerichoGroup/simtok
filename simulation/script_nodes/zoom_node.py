
import math
import threading

import omni.usd
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from pxr import UsdGeom
from std_msgs.msg import Float32

from config import get_config

_cfg = get_config()
_zoom = _cfg.zoom
_sensor = _cfg.zoom_sensor
_hfov = _cfg.zoom_hfov
_logging = _cfg.zoom_logging

CAMERA_PRIM_PATH = _zoom["camera_prim_path"]
ZOOM_TOPIC = _zoom["zoom_topic"]
ZOOM_CURVE = _zoom["zoom_curve"]        # "geometric" | "linear"
SENSOR_WIDTH_MM = _sensor["width_mm"]
SENSOR_HEIGHT_MM = _sensor["height_mm"]
HFOV_WIDE_DEG = _hfov["wide_deg"]       # at zoom = 0.0
HFOV_TELE_DEG = _hfov["tele_deg"]       # at zoom = 1.0
ZOOM_LOG_PATH = _logging["log_path"]
LOG_EVERY_S = _logging["log_every_sec"]  # heartbeat period when nothing is changing


def _focal_length_for_hfov(hfov_deg, aperture_mm=SENSOR_WIDTH_MM):
    """Focal length (mm) that produces `hfov_deg` on this sensor."""
    half = math.radians(float(hfov_deg)) * 0.5
    return float(aperture_mm) / (2.0 * math.tan(half))


def _hfov_for_focal_length(focal_mm, aperture_mm=SENSOR_WIDTH_MM):
    """Inverse of _focal_length_for_hfov: horizontal FoV (deg) at `focal_mm`."""
    return math.degrees(2.0 * math.atan(
        float(aperture_mm) / (2.0 * float(focal_mm))))


def _focal_length_mm(zoom):
    """Focal length (mm) for a zoom command in 0..1. THE lens model.

    Both sides of the system must agree on this function, or your code will
    believe it commanded a field of view that the render does not show.
    """
    z = max(0.0, min(1.0, float(zoom)))
    f_wide = _focal_length_for_hfov(HFOV_WIDE_DEG)
    f_tele = _focal_length_for_hfov(HFOV_TELE_DEG)
    if ZOOM_CURVE == "linear":
        return f_wide + (f_tele - f_wide) * z
    return f_wide * ((f_tele / f_wide) ** z)        # geometric


def _hfov_deg(zoom):
    """Horizontal field of view (deg) at a zoom command in 0..1."""
    return _hfov_for_focal_length(_focal_length_mm(zoom))


def _magnification(zoom):
    """Zoom relative to fully wide, e.g. 9.0 means 9x. Handy for logs."""
    return _hfov_deg(0.0) / max(1e-9, _hfov_deg(zoom))


def _log(msg):
    """Print AND append to ZOOM_LOG_PATH. Never raises.

    Isaac captures stdout, so the file is usually the only place you will
    actually see these. NEVER call this at module level — see THE ONE RULE.
    """
    import time                                   # local: keeps module level bare
    line = "[zoom] " + str(msg)
    try:
        print(line, flush=True)
    except Exception:
        pass
    try:
        with open(ZOOM_LOG_PATH, "a") as fh:
            fh.write(time.strftime("%H:%M:%S") + " " + line + "\n")
    except Exception:
        pass


class ZoomSubscriber:
    """Keeps the most recent zoom value from ZOOM_TOPIC."""

    def __init__(self, topic="/simtok/zoom", node_name="zoom_script_node"):
        self.topic = topic
        self.node = rclpy.create_node(node_name)
        try:
            self.node.declare_parameter("use_sim_time", True)
        except Exception:
            pass                                  # already declared: fine
        self.qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.zoom = None                          # None until the first message
        self.n_msgs = 0
        self._sub = None
        self._spinning = False

    def subscribe(self):
        if self._sub is None:
            self._sub = self.node.create_subscription(
                Float32, self.topic, self._callback, self.qos)
        if not self._spinning:
            threading.Thread(target=self._spin, daemon=True).start()
            self._spinning = True

    def _callback(self, msg):
        self.zoom = float(msg.data)
        self.n_msgs += 1

    def _spin(self):
        ex = MultiThreadedExecutor()
        ex.add_node(self.node)
        ex.spin()

    def destroy(self):
        try:
            self.node.destroy_node()
        except Exception:
            pass


def setup(db):
    """Called once when the graph starts."""
    _log("setup() starting")

    if not rclpy.ok():
        try:
            rclpy.init()
        except Exception as e:
            _log("FATAL: rclpy.init() failed: %s" % e)
            raise RuntimeError("zoom: rclpy init failed") from e

    st = db.internal_state
    st.focal_attr = None
    st.last_zoom = None
    st.last_report = 0.0

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
    if not prim.IsValid():
        # Not fatal: keep the node alive so the heartbeat can keep saying why.
        _log("FATAL: camera prim not found at '%s' — zoom cannot be applied"
             % CAMERA_PRIM_PATH)
    else:
        cam = UsdGeom.Camera(prim)

        cam.GetHorizontalApertureAttr().Set(float(SENSOR_WIDTH_MM))
        cam.GetVerticalApertureAttr().Set(float(SENSOR_HEIGHT_MM))
        cam.GetFocalLengthAttr().Set(float(_focal_length_mm(0.0)))  # start wide
        st.focal_attr = cam.GetFocalLengthAttr()
        _log("camera ready — aperture %.3f x %.3f mm, "
             "FL %.2f mm (HFoV %.1f deg) .. %.2f mm (HFoV %.1f deg), %.1fx"
             % (SENSOR_WIDTH_MM, SENSOR_HEIGHT_MM,
                _focal_length_mm(0.0), _hfov_deg(0.0),
                _focal_length_mm(1.0), _hfov_deg(1.0), _magnification(1.0)))

    st.zoom_subscriber = ZoomSubscriber(topic=ZOOM_TOPIC)
    st.zoom_subscriber.subscribe()
    _log("setup complete — subscribed to %s" % ZOOM_TOPIC)


def compute(db):
    """Called every frame. Must be cheap and must never raise."""
    import time

    st = db.internal_state
    sub = getattr(st, "zoom_subscriber", None)
    if sub is None:
        return True                               # setup() did not complete
    zoom = sub.zoom


    now = time.time()
    if now - getattr(st, "last_report", 0.0) > LOG_EVERY_S:
        st.last_report = now
        if st.focal_attr is None:
            _log("WARNING: no camera prim at '%s' — zoom cannot be applied"
                 % CAMERA_PRIM_PATH)
        elif zoom is None:
            _log("WARNING: no messages on %s yet — is anything publishing?"
                 % ZOOM_TOPIC)
        else:
            _log("alive — zoom=%.3f applied=%s msgs=%d"
                 % (zoom, st.last_zoom, sub.n_msgs))


    if zoom is None or zoom == st.last_zoom or st.focal_attr is None:
        return True

    zoom = max(0.0, min(1.0, float(zoom)))
    focal = _focal_length_mm(zoom)
    try:
        st.focal_attr.Set(float(focal))
    except Exception as e:
        _log("ERROR setting focalLength: %s" % e)
        return True

    st.last_zoom = zoom
    _log("zoom=%.3f  mag=%.2fx  FL=%.2f mm  HFoV=%.2f deg"
         % (zoom, _magnification(zoom), focal, _hfov_deg(zoom)))
    return True


def cleanup(db):
    """Called on teardown. Must not raise."""
    st = db.internal_state
    sub = getattr(st, "zoom_subscriber", None)
    if sub is not None:
        sub.destroy()
    st.zoom_subscriber = None
    st.focal_attr = None
    st.last_zoom = None
