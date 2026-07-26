import cv2
import numpy as np
import argparse
from pathlib import Path

class Config:

    WHITE_THRESHOLD = 250

    GAUSSIAN_NOISE_STD = 2.0
    TEMPORAL_NOISE_STD = 1.5
    FIXED_PATTERN_STD = 2.0

    BLUR_SIGMA = 0.8

    HOT_PIXEL_RATE = 0.00015
    DEAD_PIXEL_RATE = 0.00010

    ENABLE_FIXED_PATTERN = True
    ENABLE_TEMPORAL_NOISE = True
    ENABLE_GAUSSIAN_NOISE = True
    ENABLE_BLUR = True
    ENABLE_AGC = False
    ENABLE_LOW_RESOLUTION = False

    CONTRAST = 1.0
    BRIGHTNESS = 0


class VideoReader:

    def __init__(self, filename):
        self.cap = cv2.VideoCapture(filename)

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open {filename}")

    @property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    @property
    def width(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class VideoWriter:

    def __init__(self, filename, fps, width, height):

        self.writer = cv2.VideoWriter(
            filename,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
            False,
        )

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()


class ThermalBlur:

    def process(self, image):

        if not Config.ENABLE_BLUR:
            return image

        return cv2.GaussianBlur(image, (0, 0), Config.BLUR_SIGMA)


class FixedPatternNoise:

    def __init__(self, width, height):

        column_noise = np.random.normal(0, 2, width)

        row_noise = np.random.normal(0, 1, height)

        pixel_noise = np.random.normal(
            0,
            Config.FIXED_PATTERN_STD,
            (height, width),
        )

        self.pattern = (
            pixel_noise
            + column_noise[np.newaxis, :]
            + row_noise[:, np.newaxis]
        ).astype(np.float32)

    def process(self, image):

        if not Config.ENABLE_FIXED_PATTERN:
            return image

        self.pattern += np.random.normal(
            0,
            0.002,
            self.pattern.shape,
        )

        return image + self.pattern    


class GaussianNoise:

    def process(self, image):

        if not Config.ENABLE_GAUSSIAN_NOISE:
            return image

        image = np.clip(image, 0, 255)

        sigma = (
            Config.GAUSSIAN_NOISE_STD
            + (image / 255.0) * 4.0
        )

        noise = np.random.normal(
            0,
            sigma,
        )

        return image + noise


class TemporalNoise:

    def process(self, image):

        if not Config.ENABLE_TEMPORAL_NOISE:
            return image

        noise = np.random.normal(
            0,
            Config.TEMPORAL_NOISE_STD,
            image.shape,
        )

        return image + noise


class HotPixels:

    def __init__(self, width, height):

        self.mask = (
            np.random.rand(height, width)
            < Config.HOT_PIXEL_RATE
        )

    def process(self, image):

        image[self.mask] += 35

        return image

    
class SensorDrift:

    def __init__(self):

        self.offset = 0.0

    def process(self, image):

        self.offset += np.random.normal(0, 0.02)

        return image + self.offset


class DeadPixels:

    def __init__(self, width, height):

        self.mask = (
            np.random.rand(height, width)
            < Config.DEAD_PIXEL_RATE
        )

    def process(self, image):

        image[self.mask] = 0

        return image


class AGC:

    def process(self, image):

        if not Config.ENABLE_AGC:
            return image

        return cv2.normalize(
            image,
            None,
            0,
            255,
            cv2.NORM_MINMAX,
        )


class LowResolution:

    def process(self, image):

        if not Config.ENABLE_LOW_RESOLUTION:
            return image

        h, w = image.shape

        small = cv2.resize(
            image,
            (w // 2, h // 2),
            interpolation=cv2.INTER_AREA,
        )

        return cv2.resize(
            small,
            (w, h),
            interpolation=cv2.INTER_LINEAR,
        )


class ThermalProcessor:

    def __init__(self, width, height):

        self.blur = ThermalBlur()
        self.fixed = FixedPatternNoise(width, height)
        self.gaussian = GaussianNoise()
        self.temporal = TemporalNoise()
        self.lowres = LowResolution()
        self.agc = AGC()
        self.drift = SensorDrift()
        self.hot = HotPixels(width, height)
        self.dead = DeadPixels(width, height)

        

    def process(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # image = self.blur.process(image)

        # image = self.fixed.process(image)

        image = self.gaussian.process(image)

        # image = self.temporal.process(image)

        # image = self.lowres.process(image)

        # image = self.agc.process(image)

        # image = self.drift.process(image)

        # image = self.hot.process(image)

        # image = self.dead.process(image)

        image = image * Config.CONTRAST + Config.BRIGHTNESS

        image = np.clip(image, 0, 255)

        return image.astype(np.uint8)


class ThermalVideoPipeline:

    def run(self):

        reader = VideoReader(Config.INPUT_VIDEO)

        writer = VideoWriter(
            Config.OUTPUT_VIDEO,
            reader.fps,
            reader.width,
            reader.height,
        )

        processor = ThermalProcessor(
            reader.width,
            reader.height,
        )

        frame_count = 0

        while True:

            ret, frame = reader.read()

            if not ret:
                break

            output = processor.process(frame)

            writer.write(output)

            frame_count += 1

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")

        reader.release()
        writer.release()

        print(f"Done. Saved to {Config.OUTPUT_VIDEO}")



def main():
    parser = argparse.ArgumentParser(
        description="Apply realistic thermal camera noise to a video."
    )

    parser.add_argument(
        "video",
        type=str,
        help="Path to the input video.",
    )

    args = parser.parse_args()

    input_path = Path(args.video)

    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} does not exist.")

    Config.INPUT_VIDEO = str(input_path)

    Config.OUTPUT_VIDEO = str(
        input_path.with_name(
            f"{input_path.stem}_with_noise{input_path.suffix}"
        )
    )

    print(f"Input : {Config.INPUT_VIDEO}")
    print(f"Output: {Config.OUTPUT_VIDEO}")

    ThermalVideoPipeline().run()


if __name__ == "__main__":
    main()