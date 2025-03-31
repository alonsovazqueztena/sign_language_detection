# run.py
# Daniel Saravia
# SWE-452: Software Development Life Cycle (SDLC) II
# March 30, 2025
# I used source code from the following 
# websites to complete this assignment:
# https://chatgpt.com/share/67e9d589-79a0-8012-884c-6ca13fb5fc3a
import logging
import cv2 as cv

# Import the necessary class for the program
from frame_pipeline import FramePipeline

def main():
    # Configure logging for debugging and system condition tracking.
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Initialize the frame pipeline with the desired settings.
    pipeline = FramePipeline(
        capture_device=0,
        frame_width=1920,
        frame_height=1080,
        target_width=1920,
        target_height=1080,
        model_path="sign_language_detector.pt",
        confidence_threshold=0.5
    )

    # Run the frame pipeline.
    pipeline.run()

if __name__ == "__main__":
    main()
