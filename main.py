from isaac_core_dev_kit.isaac_manager.host_isaac_manager import HostIsaacManager
from isaac_core_dev_kit.udp.one_point_sender import OnePointSender
from isaac_core_dev_kit.udp.udp_bot import UdpBot
import isaac_core_dev_kit.dev_utils as core_utils
from isaac_core_dev_kit.core_capture.video_capture import VideoCapture

from time import sleep


def capture_first_pov(camera: UdpBot, video_capturer: VideoCapture) -> None:
    """Captures a video from the first point of view (POV) of the camera."""

    camera.move_to_point(32.12345, 35.12345, 20.0, 0.0, -90.0, 0.0, look_at_target=False)
    sleep(1)  # Wait for the camera to move to the new position
    video_capturer.start_capture()
    sleep(5)  # Record video for 5 seconds
    video_capturer.stop_capture()
    video_capturer.save_data_to("./data/test_video_1_pov1.mp4", 36)

def capture_second_pov(camera: UdpBot, video_capturer: VideoCapture) -> None:
    """Captures a video from the second point of view (POV) of the camera."""

    camera.move_to_point(32.12345, 35.12345, 50.0, 0.0, -90.0, 0.0, look_at_target=False)
    sleep(1)  # Wait for the camera to move to the new position
    video_capturer.start_capture()
    sleep(5)  # Record video for 5 seconds
    video_capturer.stop_capture()
    video_capturer.save_data_to("./data/test_video_1_pov2.mp4", 36)


def main() -> None:
    """Main function to run the simulation and capture videos."""

    core_utils.safe_rclpy_init()

    isaac_sim = HostIsaacManager(usd_path="/home/user/clones/simtok/usd/maps/scenes/cube.usda",
                      com_udp=True,
                      bbox_publisher=True,
                      sat=True,
                      core_path=".",
                      show_isaac_logs=False)
    camera = UdpBot(32.12345, 35.12345, 7.0, 0.0, -90.0, 0.0)
    video_capturer = VideoCapture()

    with isaac_sim:
        sleep(5) # Wait for scene to render
        
        video_capturer.spin()

        capture_first_pov(camera, video_capturer)
        capture_second_pov(camera, video_capturer)

    core_utils.safe_rclpy_shutdown()


if __name__ == "__main__":
    main()
