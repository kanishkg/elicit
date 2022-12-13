"""
camera.py

Simple wrapper for recording image & depth information from a RealSense camera; native resolution for the RealSense is
640 x 480. We resize to 160 x 120 (4x reduction).

Note: By default pyrealsense only builds "easily" on Linux/Windows; if you're a Mac OS user, read and edit this file,
but don't expect to be able to run things!
"""
from typing import Tuple, Union

import cv2
import numpy as np
import pyrealsense2 as rs


class Camera:
    def __init__(self, effective_resolution: Tuple[int, int] = (160, 120), use_depth: bool = False) -> None:
        """Initializes the camera, and flushes the frame buffer in preparation to read inputs."""
        self.effective_resolution, self.use_depth = effective_resolution, use_depth

        # Initialize pyrealsense2 Pipeline...
        self.pipeline = rs.pipeline()

        # Configure Streams --> Base Resolution is hardcoded at 640 x 480, 30 FPS
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)

        # Start Streaming and create Frame Aligner
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # Flush buffer...
        for _ in range(100):
            self.pipeline.wait_for_frames()

    @staticmethod
    def resize(img) -> np.ndarray:
        return cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)

    def get_frame(self) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        # Get RGB Input
        color_frame = aligned_frames.get_color_frame()
        color_image = self.resize(np.asanyarray(color_frame.get_data()))

        if self.use_depth:
            # Get Depth Image & Return
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth_image = self.resize(np.asanyarray(aligned_depth_frame.get_data()))
            return color_image, depth_image

        return color_image
