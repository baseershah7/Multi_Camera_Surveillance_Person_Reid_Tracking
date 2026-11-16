# MultiCam-Surveillance: Real-Time Person Re-Identification & Multi-Camera Tracking

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://ultralytics.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Academic Capstone Project — Bachelor of Science in Computer Science**  
> **Institution:** COMSATS University Islamabad, Wah Campus  
> **Authors:** Baseer Hassan, Fazal Ur Rehman  
> **Supervisor:** Dr. Saeed-Ur-Rehman  
> 📄 **Official Bachelor Thesis Report:** [Download / View Full PDF (57 Pages)](./Bachelor_Thesis_Report.pdf)

---

## 🎯 Overview

**MultiCam-Surveillance** is an end-to-end, real-time multi-camera person tracking and cross-camera re-identification (ReID) framework. Designed to address critical child safety and surveillance challenges in complex indoor facilities, the system processes concurrent video feeds, tracks moving subjects within individual camera views, and re-identifies individuals across disconnected camera views with spatial-temporal consistency.

https://github.com/user-attachments/assets/00e747d9-1b37-4097-b8d2-adac87326792

---

## ✨ Key Features

- **Concurrent Multi-Camera Ingestion:** Thread-safe parallel processing of up to 4 simultaneous video or RTSP camera streams.
- **Robust Object Detection:** Real-time pedestrian detection powered by Ultralytics **YOLOv8**.
- **Intra-Camera Tracking:** Continuous target tracking using **DeepSort** with Kalman filtering to handle brief intra-camera occlusions.
- **Spatio-Temporal Re-Identification (ST-ReID):** Feature extraction utilizing deep residual networks (TorchReID / ResNet-50) combined with hybrid cosine-Euclidean distance metrics and temporal consistency constraints.
- **Persistent Identity Management:** Fast disk serialization (`.npz`) and retrieval of feature embeddings across monitoring sessions.
- **Interactive Multi-Camera GUI:** Native Tkinter-based control dashboard featuring:
  - Multi-stream video preview panels
  - Dynamic matching threshold slider (0.0 - 1.0)
  - Manual ID correction, renaming, and persistent profile storage
- **Adaptive Execution:** Automatic hardware fallback between CUDA-accelerated GPU pipelines and optimized CPU execution.

---

## 🏗️ System Architecture

```text
[ Camera Stream 1 ] ──┐
[ Camera Stream 2 ] ──┼──> [ YOLOv8 Object Detection ] ──> [ DeepSort Intra-Camera Tracking ]
[ Camera Stream 3 ] ──┤                    │                                  │
[ Camera Stream 4 ] ──┘              (Bounding Boxes)                 (Tracklets & Crops)
                                                                              │
                                                                              ▼
                                                                [ TorchReID Deep Feature Extraction ]
                                                                              │
                                                                   (512D / 2048D Embeddings)
                                                                              │
                                                                              ▼
                                                                [ Global Association & ST-ReID ]
                                                                ├── Cosine & Euclidean Matching
                                                                ├── Spatial-Temporal Constraints
                                                                └── Linear Sum Assignment (Hungarian)
                                                                              │
                                                                              ▼
                                                                [ Multi-Camera GUI Dashboard ]
                                                                ├── Cross-Camera Track Synchronization
                                                                └── Feature Storage & Identity Manager
```

---

## 📁 Repository Structure

```bash
Multi_Camera_Surveillance_Person_Reid_Tracking/
├── config.yaml               # Centralized configuration (thresholds, model weights, paths)
├── main.py                   # Application launch script
├── Bachelor_Thesis_Report.pdf# Complete 57-page academic dissertation report
├── reid/                     # Re-Identification subsystem
│   ├── st_reid.py            # Spatio-temporal ReID core engine & distance computation
│   ├── feature_storage.py    # Persistent feature embedding serialization (.npz)
│   └── utils.py              # Visual bounding box and labeling utilities
├── tracking/                 # Stream processing and tracking subsystem
│   └── video_processor.py    # Multi-threaded video ingest, YOLOv8 detection & DeepSort
├── ui/                       # Graphical user interface
│   └── multi_camera_gui.py   # Multi-camera interactive Tkinter dashboard
└── README.md                 # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA support recommended (CPU execution supported)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/baseershah7/Multi_Camera_Surveillance_Person_Reid_Tracking.git
   cd Multi_Camera_Surveillance_Person_Reid_Tracking
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install torch torchvision opencv-python ultralytics deep-sort-realtime pillow pyyaml scipy torchreid
   ```

4. **Download pre-trained detection weights:**
   ```bash
   mkdir -p models
   wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

---

## 💻 Usage

Launch the multi-camera monitoring interface:
```bash
python main.py
```

### GUI Operations:
1. **Input Folder:** Select the directory containing input video feeds (or RTSP stream references).
2. **Output Folder:** Specify the directory where annotated tracking results will be recorded.
3. **Threshold Slider:** Dynamically tune ReID matching sensitivity (default: 0.6).
4. **ID Renaming:** Click on any tracked person ID to assign a known identity.
5. **Save / Load Gallery:** Persist recognized identity embeddings for continuous monitoring sessions.

---

## ⚙️ Configuration

System parameters can be adjusted via `config.yaml`:

```yaml
reid:
  alpha: 0.5                  # Weight balance between visual and spatio-temporal distance
  threshold: 0.6              # Cosine distance cutoff for identity re-assignment
  temporal_frames: 3          # Temporal smoothing window
  max_gallery_size: 100       # Maximum stored identities in active gallery

tracker:
  max_age: 30                 # DeepSort tracklet memory (frames)
  n_init: 3                   # Confirmation frames before track confirmation

models:
  yolo: "models/yolov8n.pt"   # Object detection backbone
  reid_gpu: "resnet50"        # High-accuracy GPU feature extractor
  reid_cpu: "osnet_x0_25"     # Lightweight CPU fallback feature extractor
  reid_num_classes: 751       # Market-1501 benchmark classes
```

---

## 📊 Methodology & Training

### 1. Re-Identification Feature Extractor (TorchReID)
Trained on standard pedestrian benchmarks (e.g., Market-1501, DukeMTMC-reID) using cross-entropy and triplet loss objectives:
```python
import torchreid

datamanager = torchreid.data.ImageDataManager(
    root='data/reid',
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

engine.run(save_dir='log/resnet50', max_epoch=60, eval_freq=10)
```

### 2. Pedestrian Detector Fine-Tuning (YOLOv8)
```python
from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')
results = model.train(
    data='data/pedestrian.yaml',
    epochs=100,
    imgsz=640,
    device=0,
    batch=16
)
```

---

## 📜 Academic Citation

If you reference this work or system in academic research, please cite:

```bibtex
@bachelorthesis{hassan2025multicam,
  title={Real-Time Surveillance and Tracking System for Child Safety and Engagement in a Fun House},
  author={Hassan, Baseer and Rehman, Fazal Ur},
  school={COMSATS University Islamabad, Wah Campus},
  year={2025},
  type={Bachelor of Science in Computer Science Thesis},
  note={Supervised by Dr. Saeed-Ur-Rehman}
}
```

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

---

## 📬 Contact & Author

* **Baseer Hassan** — [GitHub](https://github.com/baseershah7) • [Email](mailto:baseerhassan512@gmail.com)  
* Department of Computer Science, COMSATS University Islamabad, Wah Campus
