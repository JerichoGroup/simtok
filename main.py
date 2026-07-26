from isaac_core_dev_kit.isaac_manager.host_isaac_manager import HostIsaacManager
from isaac_core_dev_kit.udp.one_point_sender import OnePointSender
from isaac_core_dev_kit.udp.udp_bot import UdpBot
import isaac_core_dev_kit.dev_utils as core_utils
from isaac_core_dev_kit.core_capture.video_capture import VideoCapture
from isaac_core_dev_kit.core_capture.bbox_capture import BboxCapture
from isaac_core_dev_kit.core_capture.pose_capture import PoseCapture

from time import sleep


def capture_first_pov(pose_capturer: PoseCapture, bbox_capturer: BboxCapture, camera: UdpBot, video_capturer: VideoCapture) -> None:
    """Captures a video from the first point of view (POV) of the camera."""

    camera.move_to_point(32.123269, 35.1232421, 0.0, 0.0, 0.0, 0.0, look_at_target=False)
    camera.turn_to_point(32.12345, 35.12345, 0.0)
    camera.move_up_down(5) #goes up 5 meters
    sleep(0.5)  # Wait for the camera to move to the new position
    camera.turn_pitch(-5) #turns down 10 degrees
    sleep(2)  # Wait for the camera to move to the new position
    
    video_capturer.start_capture()
    pose_capturer.start_capture()
    bbox_capturer.start_capture()
    sleep(60)  # Record video for 60 seconds
    
    bbox_capturer.stop_capture()
    bbox_capturer.save_data_to("./data/bboxes/test_bbox_1.pkl")

    pose_capturer.stop_capture()
    pose_capturer.save_data_to("./data/poses/test_pose_1.pkl")

    video_capturer.stop_capture()
    video_capturer.save_data_to("./data/videos/test_video_pov_1.mp4", 36)


def capture_second_pov(pose_capturer: PoseCapture, bbox_capturer: BboxCapture, camera: UdpBot, video_capturer: VideoCapture) -> None:
    """Captures a video from the second point of view (POV) of the camera."""

    camera.move_to_point(32.1234500, 35.1233561, 0.0, 0.0, 0.0, 0.0, look_at_target=False)
    camera.turn_to_point(32.12345, 35.12345, 0.0)
    sleep(2)  # Wait for the camera to move to the new position
    video_capturer.start_capture()
    pose_capturer.start_capture()
    bbox_capturer.start_capture()
    
    sleep(60)  # Record video for 60 seconds
    
    bbox_capturer.stop_capture()
    bbox_capturer.save_data_to("./data/bboxes/test_bbox_2.pkl")

    pose_capturer.stop_capture()
    pose_capturer.save_data_to("./data/poses/test_pose_2.pkl")

    video_capturer.stop_capture()
    video_capturer.save_data_to("./data/videos/test_video_pov_2.mp4", 36)


def main() -> None:
    """Main function to run the simulation and capture videos."""

    core_utils.safe_rclpy_init()

    isaac_sim = HostIsaacManager(usd_path="/home/user/clones/simtok/usd/maps/scenes/cube.usda",
                      com_udp=True,
                      bbox_publisher=True,
                      sat=True,
                      core_path=".",
                      show_isaac_logs=False)
   
    camera = UdpBot(32.12345, 35.12345, 0.0, 0.0, 0.0, 0.0)
    video_capturer = VideoCapture()
    bbox_capturer = BboxCapture()
    pose_capturer = PoseCapture()

    with isaac_sim:
        sleep(5) # Wait for scene to render

        bbox_capturer.spin()
        video_capturer.spin()
        pose_capturer.spin()
        capture_first_pov(pose_capturer, bbox_capturer, camera, video_capturer)
        capture_second_pov(pose_capturer, bbox_capturer, camera, video_capturer)

    video_capturer.shutdown()
    bbox_capturer.shutdown()
    core_utils.safe_rclpy_shutdown()


if __name__ == "__main__":
    main()
