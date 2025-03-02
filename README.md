# Sign Language Detector

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)
- [Maintainers](#maintainers)
- [Citations and Acknowledgements](#citations-and-acknowledgements)

## Project Overview
A sign language detector application using computer vision and YOLO. Predicts letters in sign language captured from a webcam frame. Designed for Windows and server purposes.

## Key Features
- Webcam streaming (1280x720)
- Friendly, simple user interface
- Letter prediction using YOLOv11n

## System Architecture
```
.
├── images
│   ├── sign_language_test_1.jpg      # Sample test image
│   └── sign_language_test_2.jpg      # Sample test image
├── src
│   ├── ai_model_interface.py         # AI model wrapper
│   ├── detection_processor.py        # Filters/processes AI detections
│   ├── sign_language_detector_ai.pt  # YOLOv11n sign language detector model
│   ├── frame_pipeline.py             # Main processing workflow
│   ├── frame_processor.py            # Frame resizing/normalization
│   ├── main.py                       # Program execution code
│   ├── tracking_system.py            # Object tracking implementation
│   └── video_stream_manager.py       # Webcam input handling
└── test
    ├── test_ai_model_interface.py    # Unit test for AI model interface
    ├── test_ai.py                    # Detection test for AI model
    ├── test_detection_processor.py   # Unit test for detection processor
    ├── test_frame_processor.py       # Unit test for frame processor
    ├── test_tracking_system.py       # Unit test for tracking system
    └── test_video_stream_manager.py  # Unit test for video stream manager
```

## Installation

### Prerequisites
- Python 3.11.1+
- Windows 11 or Ubuntu 20.04
- Computer Webcam

```bash
# Clone repository
git clone https://github.com/alonsovazqueztena/sign_language_detection.git
cd sign_language_detection

# Create virtual environment
python -m venv env
source env/bin/activate      # Linux/MacOS
source env/Scripts/activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Operation
```bash
# Execute the program using the source code (must be in src folder)
python main.py
```

## Testing
Run validation tests in test folder:

```bash
# Run all unit tests of the program.
pytest

# Run an AI model detection test of image.

python test_ai.py   # Results in runs folder
```

## Contributing

1. Fork the repository
2. Create a feature branch:
```bash
git checkout -b feature/new-tracker
```
3. Add tests for new functionality
4. Submit a pull request

### Coding Standards
- PEP8 compliance
- Type hints for public methods
- Docstrings for all modules
- 80%+ test coverage

## License
MIT License - See LICENSE for details

## Maintainers
- Alonso Vazquez Tena  
- Daniel Saravia  
- Jason Broom

## Citations and Acknowledgements
**YOLOv11n**  
> Glenn Jocher and Jing Qiu.  
> *Ultralytics YOLO11, version 11.0.0 (2024)*  
> [GitHub Repository](https://github.com/ultralytics/ultralytics)  
> ORCID: 0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069  
> License: AGPL-3.0