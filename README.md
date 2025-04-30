# MultiCam-Surveillance: Person Tracking and Re-Identification System

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

https://github.com/user-attachments/assets/00e747d9-1b37-4097-b8d2-adac87326792

A robust multi-camera person re-identification system combining YOLOv8, DeepSort, and custom ReID models for real-time tracking across multiple video streams.

---

## 🌟 Key Features
- **Multi-Camera Tracking**: Process up to 4 simultaneous video streams
- **Spatiotemporal ReID**: Hybrid cosine/euclidean distance matching with temporal consistency
- **Real-Time Processing**: GPU-accelerated with fallback to CPU-optimized models
- **Persistent Identities**: Save/load person features across sessions
- **Interactive GUI**: Tkinter-based interface with:
  - Threshold adjustment (0.0-1.0)
  - ID renaming and management
  - Video preview windows
- **Adaptive Architecture**: Automatically selects optimal models for hardware

---

## 🛠️ Project Structure
```bash
project/
├── config.yaml # Configuration settings
├── main.py # Application entry point
├── reid/ # ReID module
│ ├── st_reid.py # Core ReID logic
│ └── feature_storage.py # Feature persistence
├── tracking/ # Tracking module
│ └── video_processor.py # Stream processing
├── ui/ # User interface
│ └── multi_camera_gui.py # Tkinter GUI
└── requirements.txt # Dependencie
```

## 🚀 Getting Started

### Installation
```bash
# Clone repository
git clone https://github.com/yourname/MultiCam-ReID.git
cd MultiCam-ReID

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pre-trained models
mkdir models
wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt

```
## GUI Controls

Select input/output folders
Adjust matching threshold (0.0-1.0)
Start/stop processing
Rename tracked IDs
Save/load identity features

##🎥 Video Demonstrations








## 🤖 Model Training
```python
# ReID Model Training
import torchreid

datamanager = torchreid.data.ImageDataManager(
    root='path/to/dataset',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size=32
)

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer='adam',
    lr=0.0003
)

engine.run(
    save_dir='log/resnet50',
    max_epoch=60,
    eval_freq=10
)
import torchreid

datamanager = torchreid.data.ImageDataManager(
    root='path/to/dataset',
    sources='market1501',
    targets='market1501',
    height=256,
    width=128,
    batch_size=32
)

model = torchreid.models.build_model(
    name='resnet50',
    num_classes=datamanager.num_train_pids,
    loss='softmax',
    pretrained=True
)

engine = torchreid.engine.ImageSoftmaxEngine(
    datamanager,
    model,
    optimizer='adam',
    lr=0.0003
)

engine.run(
    save_dir='log/resnet50',
    max_epoch=60,
    eval_freq=10
)

# YOLOv8 Training

from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    device=0,
    batch=16
)
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='path/to/data.yaml',
    epochs=100,
    imgsz=640,
    device=0,
    batch=16
)
```
## 🔧 Configuration
Edit config.yaml to modify:
```python
# config.yaml
reid:
  threshold: 0.6
  max_gallery_size: 100
tracking:
  max_age: 30
processing:
  device: cuda
input_folder: "path/to/videos"
output_folder: "path/to/results"
threshold: 0.6
max_gallery_size: 100
temporal_frames: 3
frame_interval: 0.033
```

## 📝 License
This project is MIT licensed. See LICENSE for details.

## 🤝 Contributing
Fork repository
Create feature branch (git checkout -b feature/amazing-feature)
Commit changes (git commit -m 'Add some amazing feature')
Push branch (git push origin feature/amazing-feature)
Open pull request

## 📧 Contact
Baseer Hassan - baseerhassan512@gmail.com

## 🙏 Acknowledgments
Ultralytics for YOLOv8
TorchReID for ReID models
DeepSort for tracking
