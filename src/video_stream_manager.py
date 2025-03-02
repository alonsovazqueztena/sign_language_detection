# Alonso Vazquez Tena
# STG-452: Software Development Life Cycle (SDLC) II
# March 2, 2025
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


# This class serves as code for a video stream manager.

# This serves to handle the logic to properly
# take in a video stream from a webcam.
class VideoStreamManager:
    """Creates and sets up the video stream manager."""

    # This method initializes the video stream manager.
    
    # The captured device is taken in as an index,
    # the matching width and height of the lowest available video frame
    # resolution from the webcam is also taken 
    # in, all as arguments.
    def __init__(
            self, capture_device=0, 
            frame_width=1280, frame_height=720):
        """Initialize the video stream manager.
        
        Keyword arguments:
        self -- instance of the video stream manager
        capture_device -- index of video capture stream device (default 0)
        frame_width -- width of video capture stream frame (default 1280)
        frame_height -- height of video capture stream frame (default 720)
        """
        self.capture_device = capture_device
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.capture = None

        # The logging is configured here. 
        
        # Basic info is put in, which includes the time, 
        # level name, and messages.
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
            )

    # This initializes the video stream taken in
    # the webcam.
    def initialize_stream(
            self):
        """Initialize the video stream."""
        # The log includes a message displaying that the 
        # video stream is initializing.
        logging.info(
            "The webcam stream is initializing..."
            )

        # OpenCV is executed on the video stream.
        self.capture = cv.VideoCapture(
            self.capture_device)

        # Captured frame width and height is set here.
        self.capture.set(
            cv.CAP_PROP_FRAME_WIDTH, self.frame_width
            )
        self.capture.set(
            cv.CAP_PROP_FRAME_HEIGHT, self.frame_height
            )

        # This checks if webcam can be connected to
        # and opened.
        if not self.capture.isOpened():
            logging.critical(
                "Webcam open failed."
                )
            raise RuntimeError(
                "ERROR: Cannot open the webcam."
                )

        # This message is displayed through a log that 
        # the stream has been initialized with a certain resolution.
        logging.info(
            f"The webcam stream has been initialized with resolution "
            f"{self.frame_width} by {self.frame_height}."
            )

    # This method gets a frame from the webcam and
    # returns it in the program.
    def get_frame(
            self):
        """Retrieve the frame from the webcam."""

        # If the webcam cannot be opened or 
        # there is no webcam detected,
        # an error is raised and output in a log.
        if not self.capture or not self.capture.isOpened():
            logging.error(
                "The webcam stream is not initialized."
                )
            raise RuntimeError(
                "ERROR: The webcam stream cannot be initialized."
                )

        # This reads in the frame.
        ret, frame = self.capture.read()

        # If the frame cannot be captured, an error is 
        # raised and output in a log.
        if not ret:
            logging.error(
                "Failed to capture the frame."
                )
            raise RuntimeError(
                "ERROR: The frames cannot be captured."
            )

        # If an invalid frame is received, output a warning 
        # in a log.
        if frame is None:
            logging.warning(
                "The captured frame is None (invalid)."
                )
            return None

        # When a frame is successfully received, the height, width,
        # and channels are logged. 
        logging.info(
            f"Captured frame of size: {frame.shape}"
            )
        return frame

    # This method releases the webcam resources upon
    # key from the user to terminate.
    def release_stream(
            self):
        """Release the webcam."""

        # If the webcam is opened and detected
        # (thus, a webcam exists), the webcam will end and
        # this will be output in a log.
        if self.capture and self.capture.isOpened():
            self.capture.release()
            logging.info(
                "The webcam stream was released."
                )

    # This is a simple enter method that only initializes the stream.
    def __enter__(
            self):
        """Initialize the stream."""
        self.initialize_stream()
        return self

    # This is a simple exit method that only releases the stream.
    
    # An exit message is displayed as well 
    # as the traceback through the terminal.
    def __exit__(
            self, exc_type, 
            exc_value, traceback):
        """Exit the video stream.
        
        self -- instance of the video stream manager
        exc_type -- class of the exception that occurred during execution
        exc_value -- actual exception instance that occurred
        traceback -- contains stack trace where exception was raised
        """
        self.release_stream()