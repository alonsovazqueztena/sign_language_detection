# Alonso Vazquez Tena
# STG-452: Capstone Project II
# February 3, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8
# (used as starter code for basic functionality).

# This project requires the usage of logs for the developer
# to understand the conditions of the system, whether
# an error has occurred or the execution of the class was a success.
import logging

# This project requires the usage of computer vision.

# In this case, OpenCV will be used.
import cv2 as cv

# Utilizing AI models requires the usage of arrays and matrices
# for data processing.
import numpy as np


# This class serves to process frames for the YOLO model.

# This ensures that the frames can be processed to ensure
# more accurate object detection.
class FrameProcessor:
    """Creates and sets up the frame processor."""

    # This method initializes the frame processor.
    
    # The target width and height of the frame are taken in as arguments.
    def __init__(
            self, target_width=1280, 
            target_height=720):
        """Initialize the frame processor.

        Keyword arguments:
        self -- instance of the frame processor
        target_width -- target width of the frame (default 640)
        target_height -- target height of the frame (default 640)
        """
        self.target_width = target_width
        self.target_height = target_height

        # The logging is configured here. 
        
        # Basic info is put in, which includes the time, 
        # level name, and messages.
        logging.basicConfig(
            level=logging.INFO, 
            format="%(asctime)s - %(levelname)s - %(message)s"
            )
    
    # This preprocesses a single frame for YOLO input.
    def preprocess_frame(
            self, frame):
        """Preprocess a single frame for YOLO input."""

        # If the frame is invalid or empty, an error is logged and raised.
        if frame is None or frame.size == 0:
            logging.error(
                "Invalid frame provided for preprocessing."
                )
            raise ValueError(
                "Invalid frame provided for preprocessing."
                )

        # The original frame size is logged.
        logging.info(
            f"Original frame size: {frame.shape}"
            )

        # Resize the frame to the target dimensions.
        resized_frame = cv.resize(
            frame, (self.target_width, 
            self.target_height))
        
        # This demonstrates in a log what dimension the frame was resized to.
        logging.info(
            f"Resized frame to: {self.target_width} by {self.target_height}"
            )

        # This adds a batch dimension as the AI model expects 4D input: 
        # batch, width, height, and channels.

        # We are looking at one frame at a time for the batch.
        preprocessed_frame = np.expand_dims(
            resized_frame, axis=0)
        logging.info(
            f"Added batch dimension. Preprocessed frame shape: {preprocessed_frame.shape}"
            )
        
        # We return the preprocessed frame.
        return preprocessed_frame

    # This preprocesses multiple frames for YOLO input.

    # This takes a list of frames and returns a 
    # batch of preprocessed frames.
    def preprocess_frames(
            self, frames):
        """Preprocesses multiple frames for YOLO input."""

        # If the frames are invalid or empty, an error is logged and raised.
        if not frames or not isinstance(
                frames, list):
            logging.error(
                "Invalid list of frames provided for batch preprocessing."
                )
            raise ValueError(
                "Invalid list of frames provided for batch preprocessing."
                )
        
        # The number of frames in the batch is logged.
        logging.info(
            f"Processing a batch of {len(frames)} frames."
            )

        # Preprocess each frame in the list.
        preprocessed_frames = [
            self.preprocess_frame(frame) for frame in frames
            ]

        # This combines all preprocessed frames into a single batch.
        batch_frames = np.vstack(
            preprocessed_frames)
        logging.info(
            f"Batch of preprocessed frames shape: {batch_frames.shape}"
            )

        # We return the batch of preprocessed frames.
        return batch_frames