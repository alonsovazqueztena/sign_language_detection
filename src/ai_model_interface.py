# Alonso Vazquez Tena
# SWE-452: Software Development Life Cycle (SDLC) II
# March 2, 2025
# I used source code from the following 
# website to complete this assignment:
# https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8
# (used as starter code for basic functionality).

# This project requires the usage of logs for the developer
# to understand the conditions of the system, whether
# an error has occurred or the execution of the class was a success.
import logging

# We are using the YOLOv11n model for object detection.
from ultralytics import YOLO


# This class serves to detect objects in a frame using the AI model.

# This ensures that the AI model will confidently detect objects.
class AIModelInterface:
    """Interface for the AI model to run inference 
    and process detections."""

    # This method initializes the AI model interface.

    # The model path is where our trained AI model is stored and
    # the confidence threshold is the minimum 
    # confidence score for detections.
    def __init__(
            self, model_path="sign_language_detector_ai.pt", 
            confidence_threshold=0.5):
        """
        Initializes the AI model interface.

        Keyword arguments:
            model_path -- path to the AI model file.
            confidence_threshold -- minimum confidence score for detections.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold

        ## The logging is configured here. 
        
        # Basic info is put in, which includes the time, 
        # level name, and messages.
        logging.basicConfig(
            level=logging.INFO, 
            format="%(asctime)s - %(levelname)s - %(message)s"
            )

        # This loads the AI model from the specified path.
        try:
            self.model = YOLO(
                self.model_path)
            logging.info(
                f"AI model loaded successfully from {self.model_path}"
                )
        except Exception as e:
            logging.error(
                f"Failed to load AI model from {self.model_path}: {e}"
                )
            raise

    # This method runs inference on a single frame.
    def predict(self, frame):
        """Runs inference on a single frame and extracts detections."""

        try:

            # This runs an inference on a frame.
            results = self.model.predict(
                source=frame, imgsz=640, 
                conf=self.confidence_threshold)

            # This processes the results by adding the detection to a list.
            detections = []

            for result in results:

                # This checks if any boxes are available.
                if result.boxes is not None: 
                    for box in result.boxes:

                        # This extracts a bounding box, confidence, and class ID.
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        class_id = int(
                            box.cls[0].item())
                        label = self.model.names[class_id]

                        # If a detection is above the confidence threshold,
                        # it is added to the list of detections.
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                "bbox": [x_min, y_min, x_max, y_max],
                                "confidence": confidence,
                                "class_id": class_id,
                                "label": label
                            })

            # This logs the detections.
            logging.info(f"Detections: {detections}")
            return detections

        # Any errors that occur during prediction are logged.
        except Exception as e:
            logging.error(
                f"Error during prediction: {e}"
                )
            return []

    # This method runs inference on a batch of frames.
    def predict_batch(self, frames):
        """Runs inference on a batch of frames and extract detections."""
        
        try:
            # This runs an inference on a batch of frames.
            results = self.model.predict(
                source=frames, imgsz=640, 
                conf=self.confidence_threshold)

            # This processes the results by adding the detections to a list.
            all_detections = []

            for result in results:
                detections = []

                # This checks if any boxes are available.
                if result.boxes is not None:
                    for box in result.boxes:

                        # This extracts a bounding box, confidence, and class ID.
                        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                        confidence = box.conf[0].item()
                        class_id = int(
                            box.cls[0].item())
                        label = self.model.names[class_id]

                        # If a detection is above the confidence threshold,
                        # it is added to the list of detections.
                        if confidence >= self.confidence_threshold:
                            detections.append({
                                "bbox": [x_min, y_min, x_max, y_max],
                                "confidence": confidence,
                                "class_id": class_id,
                                "label" : label
                            })

                # Any detections found in a frame is 
                # added to a batch list of detections.
                all_detections.append(detections)

            # This logs the detections in terms of batches.
            logging.info(
                f"Batch detections: {all_detections}"
                )
            return all_detections

        # Any errors that occur during batch prediction are logged.
        except Exception as e:
            logging.error(
                f"Error during batch prediction: {e}"
                )
            return []