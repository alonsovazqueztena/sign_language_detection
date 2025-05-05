# Sign Language Detector

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Web Backend Setup](#web-backend-setup)
- [Hardware Setup](#hardware-setup)
- [Model Training](#model-training)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Maintainers](#maintainers)
- [Citations and Acknowledgements](#citations-and-acknowledgements)

---

## Project Overview

![ASL alphabet picture](asl_picture.png)

A sign language detector application using computer vision and YOLO. The system predicts letters in sign language captured from a webcam frame. It is designed to run on Windows, Linux, and server environments.

---

## Key Features
- **Letter Prediction:** Uses the YOLO12m model for real-time sign language detection.
- **Webcam Streaming:** Processes live webcam input.
- **Simple User Interface:** Easy-to-use interface for demonstration purposes.
- **Web API Backend:** A Django REST Framework (DRF) API for processing image uploads (development phase).

---

## System Architecture

```
.
├── notebooks
│   └── ai_model_training.ipynb
├── src
│   ├── ai_model_interface.py
│   ├── frame_pipeline.py
│   ├── main.py
│   ├── sign_language_detector.pt
│   ├── tracking_system.py
│   └── video_stream_manager.py
├── test
│   └── test_ai.py
└── test_images
    ├── sign_language_test_1.jpg
    ├── sign_language_test_2.jpg
    ├── ...
    └── sign_language_test_10.jpg
```

Django backend:

```
backend
├── backend
│   ├── asgi.py
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   └── wsgi.py
├── db.sqlite3
├── detectionapi
│   ├── admin.py
│   ├── api_utils.py
│   ├── apps.py
│   ├── __init__.py
│   ├── migrations
│   │   └── __init__.py
│   ├── models.py
│   ├── urls.py
│   └── views.py
└── manage.py
```

---

## Installation

### Prerequisites
- Python 3.11.9
- PIP 25.0.1
- Windows 11 or Ubuntu 20.04
- Webcam or IP camera

### Steps

```bash
git clone https://github.com/alonsovazqueztena/sign_language_detection.git
cd sign_language_detection

python -m venv env
source env/bin/activate  # Linux/MacOS
# OR
env\Scripts\activate   # Windows

pip install -r requirements.txt
python3 backend/manage.py migrate
```

---

## Usage

### Basic Operation (Non-Web)
```bash
cd src
python main.py
```

### User Controls
| Key | Description |
|-----|-------------|
| q   | Quit program |

---

## Web Backend Setup

1. Ensure `src/` is in the PYTHONPATH (done in `manage.py`)
2. Start the server:
   ```bash
   python3 backend/manage.py runserver
   ```

### API Endpoint

- **URL:** `http://127.0.0.1:8000/api/detect/`
- **Method:** `POST`
- **Payload:** `image=<image file>`

```bash
curl -X POST -F "image=@test_images/sign_language_test_1.jpg" http://127.0.0.1:8000/api/detect/
```

---

## Hardware Setup

Use **Iriun Webcam** to turn your phone into a webcam:
- Install on phone + PC
- Connect phone via USB
- Launch Iriun on both devices

---

## Model Training

Notebook: `notebooks/ai_model_training.ipynb`

Use Google Colab for training and adjust paths accordingly. Enable GPU:

```python
!nvidia-smi
import torch
print(torch.cuda.is_available())
```

Train:
```python
model = YOLO("your_model.pt")
train_results = model.train(
  data="your_dataset.yaml", epochs=100, imgsz=640, device="cuda"
)
```

TensorBoard:
```python
%load_ext tensorboard
%tensorboard --logdir path/to/runs
```

---

## Testing

### Module Testing
```bash
cd src
python main.py
```

### AI Model Testing
Update test image in `test/test_ai.py`, then run:

```bash
cd test
python test_ai.py
```

Results are saved to `runs/predict`.

---

## Troubleshooting

**No frames available error?**

Check capture device index in `video_stream_manager.py`:
```python
def __init__(self, capture_device=0)
```

---

## Contributing

1. Fork repo
2. Create branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Add tests, ensure style
4. PR to main branch

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Maintainers

- Alonso Vazquez Tena – AI Engineer  
- Daniel Saravia – Cloud Engineer  
- Jason Broom – Front-End Developer

---

## Citations and Acknowledgements

**YOLOv12**
```bibtex
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
```

**Dataset**
```bibtex
@misc{asl-alphabet-recognition_dataset,
  title = {ASL Alphabet Recognition Dataset},
  author = {University of Central Florida},
  url = {https://universe.roboflow.com/university-of-central-florida/asl-alphabet-recognition},
  year = {2023}
}
```

**Project Origin**

Adapted from the [AIegis Beam project](https://github.com/alonsovazqueztena/Mini_C-RAM_Capstone)
