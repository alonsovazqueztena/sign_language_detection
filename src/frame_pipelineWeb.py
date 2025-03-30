# src/frame_pipelineWeb.py

# This allows for the asynchronous execution of the AI model
# predictions using threads.
import concurrent.futures

# This project requires the usage of logs for the developer
# to understand the conditions of the system, whether
# an error has occurred or the execution of the class was a success.
import logging

# All the classes are imported from the src folder
# to be used in the frame pipeline class.
from ai_model_interface import AIModelInterface
from detection_processor import DetectionProcessor
from frame_processor import FrameProcessor
from tracking_system import TrackingSystem
from video_stream_manager import VideoStreamManager

# This project requires the usage of computer vision.
# In this case, OpenCV will be used.
import cv2 as cv


# This class serves as a frame pipeline that 
# captures frames from a video stream,
# processes them, runs AI + detection filtering, 
# then tracks objects over time.
class FramePipeline:
    """A pipeline that captures frames from a video stream, processes them, 
    runs YOLO + detection filtering, then tracks objects over time."""

    def __init__(
        self,
        capture_device=1,
        frame_width=1920,
        frame_height=1080,
        target_width=1920,
        target_height=1080,
        model_path="sign_language_detector.pt",
        confidence_threshold=0.5,
        detection_processor=None,
        tracking_system=None
    ):
        """Initialize the frame pipeline.

        Keyword arguments:
        capture_device -- device ID for the video stream,
        frame_width -- width of the video frame,
        frame_height -- height of the video frame,
        target_width -- target width for a preprocessed frame,
        target_height -- target height for a preprocessed frame,
        model_path -- path to the AI model file,
        confidence_threshold -- minimum confidence score for detections,
        detection_processor -- instance of DetectionProcessor to filter detections,
        tracking_system -- instance of TrackingSystem to track objects.
        """
        self.video_stream = VideoStreamManager(
            capture_device=capture_device, 
            frame_width=frame_width, 
            frame_height=frame_height
        )

        self.frame_processor = FrameProcessor(
            target_width=target_width, 
            target_height=target_height
        )

        self.ai_model_interface = AIModelInterface(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )

        self.detection_processor = detection_processor or DetectionProcessor(
            target_classes=None
        )

        self.tracking_system = tracking_system or TrackingSystem(
            max_disappeared=50, 
            max_distance=50
        )

    def draw_detections(self, frame, detections):
        """Draw bounding boxes and centroids on the frame."""
        for det in detections:
            bbox = det["bbox"]
            confidence = det["confidence"]
            x_min, y_min, x_max, y_max = map(int, bbox)
            label = f"{det['label']} {confidence:.3f}"
            font = cv.FONT_HERSHEY_TRIPLEX
            font_scale = 2
            thickness = 4

            (text_width, text_height), baseline = cv.getTextSize(
                label, font, font_scale, thickness)
            margin = 5
            cv.rectangle(frame,
                (x_min, y_min - text_height - baseline - margin),
                (x_min + text_width, y_min),
                (0, 0, 0), -1)

            cv.putText(frame, label, 
                       (x_min, y_min - margin), 
                       font, font_scale, 
                       (230, 216, 173), thickness)

    def draw_tracked_objects(self, frame, tracked_objects):
        """Draws tracked object IDs and centroids."""
        for detection in tracked_objects.values():
            bbox = detection["bbox"]
            x_min, y_min, x_max, y_max = map(int, bbox)
            cx, cy = detection["centroid"]

            cv.rectangle(frame, (x_min, y_min), 
                         (x_max, y_max), (230, 216, 173), 2)
            cv.circle(frame, (int(cx), int(cy)), 
                      4, (230, 216, 173), -1)

    def run(self):
        """Captures frames, runs preprocessing + AI + detection processing,
        then updates the tracking system and displays both AI detections
        and tracked objects in real time."""
        try:
            with self.video_stream as stream, concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                cv.namedWindow("Sign Language View", cv.WINDOW_NORMAL)
                cv.resizeWindow("Sign Language View", 800, 600)
                cv.setWindowProperty("Sign Language View", cv.WND_PROP_TOPMOST, 1)

                while True:
                    frame = stream.get_frame()
                    if frame is None:
                        logging.warning("No frame captured. Exiting the pipeline.")
                        break

                    processed_frame = self.frame_processor.preprocess_frame(frame)
                    future = executor.submit(self.ai_model_interface.predict, processed_frame[0])
                    raw_detections = future.result()
                    processed_detections = self.detection_processor.process_detections(raw_detections)
                    tracked_objects = self.tracking_system.update(processed_detections)

                    self.draw_detections(frame, processed_detections)
                    self.draw_tracked_objects(frame, tracked_objects)
                    cv.imshow("Sign Language View", frame)

                    if cv.waitKey(1) & 0xFF == ord('q'):
                        break

        except Exception as e:
            logging.error(f"Error in FramePipeline run: {e}")

        finally:
            self.video_stream.release_stream()
            cv.destroyAllWindows()

    def process_image(self, frame):
        """
        Process a single image frame for detection and return the detection results.
        
        This method is intended for API usage. It:
          - Preprocesses the input image.
          - Runs the AI model prediction synchronously.
          - Processes and filters the detections.
          
        Parameters:
            frame (numpy.ndarray): The input image frame.
        
        Returns:
            dict: The processed detections in a JSON-serializable format.
        """
        # Preprocess the image. Note: preprocess_frame might return a tuple; adjust if needed.
        processed_frame = self.frame_processor.preprocess_frame(frame)
        
        # Run the model prediction (synchronously).
        raw_detections = self.ai_model_interface.predict(processed_frame[0])
        
        # Process the raw detections.
        processed_detections = self.detection_processor.process_detections(raw_detections)
        
        return processed_detections
