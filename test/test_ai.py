# Alonso Vazquez Tena | SWE-452: Software Development Life Cycle (SDLC) II | April 5, 2025
# This is my own code.

from ultralytics import YOLO # Import YOLO class for AI.

model = YOLO("..\src\sign_language_detector.pt") # Create model instance.
results = model.predict("..\\test_images\sign_language_test_1.jpg", conf=0.5, imgsz=640, show=True, save=True) # Run inference (confidence, image size, display, save prediction).
print(results[0].boxes) # Print detected bounding boxes.