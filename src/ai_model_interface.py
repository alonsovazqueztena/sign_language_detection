# Alonso Vazquez Tena | SWE-452: Software Development Life Cycle (SDLC) II | April 4, 2025
# Source: https://chatgpt.com/share/67a05526-d4d8-800e-8e0d-67b03ca451a8, https://grok.com/share/bGVnYWN5_7008140a-9936-4b7f-b83d-0760c7ea866c
# Daniel Saravia Source: https://grok.com/share/bGVnYWN5_52adc247-cde4-41e4-80bd-c70ef0c81dc9

from ultralytics import YOLO # YOLO model for object detection

class AIModelInterface:
    """Optimized interface for YOLO sign language detection."""
    def __init__(self, model_path="sign_language_detector.pt", confidence_threshold=0.5):
        """Initialize YOLO model with minimal setup."""
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

    def predict(self, frame):
        """Run inference and return minimal detection data."""
        results = self.model.predict(
            source=frame, # Input frame for detection.
            imgsz=640, # Resize frame to 640 pixels.
            conf=self.confidence_threshold, # Use defined confidence threshold.
            verbose=False # Disable logging.
        )
        detections = [] # Initialize empty list for detections
        for result in results:
            if result.boxes is None:
                continue # Skip if no boxes detected.
            for box in result.boxes:
                label = self.model.names[int(box.cls[0])] # Take in class label.
                if label not in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]:
                    continue
                bbox = box.xyxy[0].tolist() # Extract bounding box coordinates as list.
                centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2) # Calculate centroid.
                detections.append({"bbox": bbox, "centroid": centroid, "label": label}) # Append detection info.
        return detections