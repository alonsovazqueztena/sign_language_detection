# api.py

from fastapi import FastAPI, UploadFile, File, HTTPException
import numpy as np
import cv2
import uvicorn

# Import the detection pipeline from your src folder.
from src.frame_pipelineWeb import FramePipeline

app = FastAPI(
    title="Sign Language Detector API",
    description="API for processing images to detect sign language letters.",
    version="1.0"
)

# Initialize the pipeline. The capture_device is not used here.
pipeline = FramePipeline(
    capture_device=0,
    frame_width=1920,
    frame_height=1080,
    target_width=1920,
    target_height=1080,
    model_path="src/sign_language_detector.pt",
    confidence_threshold=0.5
)

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    """
    Endpoint to process an uploaded image and return detection results.
    Expects an image file (e.g., JPEG or PNG).
    """
    # Read the file bytes
    contents = await file.read()
    # Convert bytes data to a NumPy array
    nparr = np.frombuffer(contents, np.uint8)
    # Decode the image using OpenCV
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    
    # Process the image using the pipeline's process_image method.
    detections = pipeline.process_image(img)
    return {"detections": detections}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Sign Language Detector API. Visit /docs for API documentation."}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
