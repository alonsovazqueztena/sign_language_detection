# test_process_image.py

import cv2 as cv
from frame_pipelineWeb import FramePipeline  # Adjusted import for running inside src folder

def main():
    # Path to the test image from the test_images folder.
    image_path = "../test_images/sign_language_test_1.jpg"
    
    # Load the image using OpenCV.
    img = cv.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image from {image_path}")
        return

    # Initialize the pipeline.
    pipeline = FramePipeline(
        capture_device=0,  # Not used in API mode.
        frame_width=1920,
        frame_height=1080,
        target_width=1920,
        target_height=1080,
        model_path="sign_language_detector.pt",  # adjust path if needed
        confidence_threshold=0.5
    )
    
    # Process the image.
    detections = pipeline.process_image(img)
    
    # Print the detections.
    print("Detections:")
    print(detections)

if __name__ == "__main__":
    main()
