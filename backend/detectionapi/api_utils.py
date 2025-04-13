# api_utils.py
import cv2 as cv
import numpy as np
from src.ai_model_interface import AIModelInterface

# Initialize the model interface once (you may adjust confidence threshold and model path as needed)
MODEL_INTERFACE = AIModelInterface("src/sign_language_detector.pt", confidence_threshold=0.5)

def process_frame_from_buffer(file_buffer):
    """
    Process an image file given as a byte stream.
    
    Arguments:
        file_buffer (bytes): The raw bytes of the uploaded image.
    
    Returns:
        detections (list): A list of detection dictionaries.
    """
    # Convert file bytes into a NumPy array and decode the image
    np_arr = np.frombuffer(file_buffer, np.uint8)
    frame = cv.imdecode(np_arr, cv.IMREAD_COLOR)
    
    if frame is None:
        raise ValueError("Could not decode image. Ensure a proper image format was uploaded.")
    
    # Run inference using the pre-loaded model interface
    detections = MODEL_INTERFACE.predict(frame)
    
    return detections
