# Alonso Vazquez Tena
# SWE-452: Software Development Life Cycle (SDLC) II
# March 30, 2025
# This is my own code.

# We import the AI model's class here.
from ultralytics import YOLO

# We create an instance of the trained AI model here.
model = YOLO("..\src\sign_language_detector.pt")

# We use the model to infer on a test image here.

# The confidence threshold is set to 50%,
# the image size is set to 640, and the processed 
# image is displayed and saved.
results = model.predict(
    "..\\test_images\sign_language_test_6.jpg", conf=0.5, 
    imgsz=640, show=True, save=True, project="..\\runs")

# We print the bounding boxes of the detected objects here.
print(
    results[0].boxes
    )