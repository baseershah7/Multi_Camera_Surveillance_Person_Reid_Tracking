# Multi-Camera Surveillance: Real-Time Person Re-Identification & Cross-Feed Tracking

```text
========================================================================================
[PROJECT]     MultiCam-Surveillance
[AFFILIATION] COMSATS University Islamabad, Department of Computer Science
[AUTHORS]     Baseer Hassan, Fazal Ur Rehman
[SUPERVISOR]  Dr. Saeed-Ur-Rehman
[DEGREE]      Bachelor of Science in Computer Science (Final Year Capstone Project)
[REPORT]      ./Bachelor_Thesis_Report.pdf (57-page Formal Academic Dissertation)
========================================================================================
```

---

## 01. ABSTRACT & PROBLEM FORMULATION

In multi-camera surveillance environments (e.g., educational facilities, transit hubs, indoor entertainment complexes), maintaining subject identity across non-overlapping fields of view presents significant challenges due to perspective distortion, dynamic occlusions, and variable lighting.

This repository implements an end-to-end, multi-threaded computer vision pipeline capable of:
1. Concurrently capturing and processing up to 4 video/RTSP streams.
2. Localizing human subjects per frame via YOLOv8.
3. Establishing intra-camera tracklets using DeepSort (Kalman filtering + Hungarian algorithm).
4. Generating deep visual appearance embeddings (TorchReID ResNet-50 / OSNet).
5. Resolving global identity assignment across disconnected camera feeds using a spatial-temporal distance matrix and the Kuhn-Munkres (Hungarian) algorithm.
6. Serializing and managing active identity galleries (`.npz` feature vectors) in real time via an interactive desktop operator GUI.

---

## 02. PIPELINE ARCHITECTURE

```text
+-----------------------+     +-----------------------+
|  Camera Stream 01/02  | ... |  Camera Stream 03/04  |
+-----------+-----------+     +-----------+-----------+
            |                             |
            +--------------+--------------+
                           |
                           v
            +-----------------------------+
            |  Multi-Threaded Video Ingest|
            |  (thread-safe frame queues) |
            +--------------+--------------+
                           |
                           v
            +-----------------------------+
            |   YOLOv8 Pedestrian Detector|  --> [Bounding Box Localization]
            +--------------+--------------+
                           |
                           v
            +-----------------------------+
            |   DeepSort Tracklet Engine  |  --> [Intra-Camera Trajectory & Smoothness]
            +--------------+--------------+
                           | (Cropped Subject Tensors)
                           v
            +-----------------------------+
            |    TorchReID Feature Extr.  |  --> [512-D / 2048-D Normalized Embeddings]
            +--------------+--------------+
                           |
                           v
            +-----------------------------+
            | Spatio-Temporal Association |
            | D_total = alpha*D_vis +     |  --> [Kuhn-Munkres Global Matching]
            |           (1-alpha)*D_temp  |
            +--------------+--------------+
                           |
            +--------------+--------------+
            |                             |
            v                             v
+-----------------------+     +-----------------------+
| Identity Gallery Disk |     | Tkinter Operator GUI  |
| (Persistent .npz DB)  |     | (Multi-Feed Dashboard)|
+-----------------------+     +-----------------------+
```

---

## 03. SPATIO-TEMPORAL RE-IDENTIFICATION METHODOLOGY

The cross-camera matching engine combines photometric visual similarity with kinematic temporal plausibility:

```text
Visual Distance:
  D_visual(e_i, e_j) = 1 - ( (e_i . e_j) / (||e_i|| * ||e_j||) )

Spatial-Temporal Distance:
  D_st(t_i, t_j, cam_i, cam_j) = | delta_t - T_expected(cam_i, cam_j) | / sigma_t

Composite Association Matrix:
  C(i, j) = alpha * D_visual(i, j) + (1 - alpha) * D_st(i, j)

Optimal Assignment:
  argmin SUM( C(i, j) * X_ij )  subject to  X_ij in {0, 1}
```

If the minimum assignment distance exceeds `threshold` (default: 0.6), the tracklet is initialized as a novel identity in the global gallery.

---

## 04. REPOSITORY SCHEMATIC

```text
.
├── config.yaml                    # System configuration (hyperparameters, model paths)
├── main.py                        # Entry-point runner
├── Bachelor_Thesis_Report.pdf     # Complete 57-page academic dissertation report
├── structure.txt                  # Directory inventory dump
│
├── reid/
│   ├── st_reid.py                 # Core Spatio-Temporal ReID association engine
│   ├── feature_storage.py         # Disk serialization / deserialization (.npz vectors)
│   └── utils.py                   # Bounding box rendering & text labeling helpers
│
├── tracking/
│   └── video_processor.py         # Multi-threaded stream ingest, YOLOv8 & DeepSort
│
└── ui/
    └── multi_camera_gui.py        # Operator GUI (Tkinter), ID renaming & feed preview
```

---

## 05. INSTALLATION & ENVIRONMENT SETUP

### System Requirements
* Python 3.8 - 3.11
* CUDA-capable GPU (Compute Capability >= 6.0 recommended; automatic CPU fallback available)

### Quickstart

```bash
# 1. Clone repository
git clone https://github.com/baseershah7/Multi_Camera_Surveillance_Person_Reid_Tracking.git
cd Multi_Camera_Surveillance_Person_Reid_Tracking

# 2. Initialize virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install core dependencies
pip install torch torchvision opencv-python ultralytics deep-sort-realtime pillow pyyaml scipy torchreid

# 4. Fetch detection model weights
mkdir -p models
wget -O models/yolov8n.pt https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

---

## 06. EXECUTION & OPERATOR CONTROLS

Launch the centralized operator interface:
```bash
python main.py
```

### Operational Workflow
* `Input Folder`: Directory containing synchronized video streams (`.mp4`, `.avi`) or RTSP descriptors.
* `Output Folder`: Destination directory for telemetry and annotated tracking output.
* `Threshold Adjustment`: Real-time cosine cutoff calibration ($0.00$ to $1.00$).
* `Identity Override`: Direct operator manual ID renaming for cross-feed correction.
* `Persist Gallery`: Serializes active cluster centroids to disk (`.npz`) for session continuation.

---

## 07. CONFIGURATION MATRIX (`config.yaml`)

```yaml
reid:
  alpha: 0.5                  # Weighting factor between appearance and temporal distance
  threshold: 0.6              # Identity distance rejection cutoff
  temporal_frames: 3          # Temporal smoothing window
  max_gallery_size: 100       # Maximum active gallery tracks

tracker:
  max_age: 30                 # Maximum missed frames before dropping tracklet
  n_init: 3                   # Required consecutive detections before confirmation

models:
  yolo: "models/yolov8n.pt"   # Pedestrian detection backbone
  reid_gpu: "resnet50"        # High-capacity GPU ReID model
  reid_cpu: "osnet_x0_25"     # Lightweight CPU fallback ReID model
  reid_num_classes: 751       # Training classes (Market-1501 standard)

processing:
  device: "cuda"              # "cuda" or "cpu"
```

---

## 08. ACADEMIC DISSERTATION & CITATION

The theoretical derivation, empirical benchmarks across Market-1501, latency profiles, and failure case analyses are documented in full detail within the formal thesis report:

* **Full Text:** [`./Bachelor_Thesis_Report.pdf`](./Bachelor_Thesis_Report.pdf)

```bibtex
@bachelorthesis{hassan2025multicam,
  title={Real-Time Surveillance and Tracking System for Child Safety and Engagement in a Fun House},
  author={Hassan, Baseer and Rehman, Fazal Ur},
  school={COMSATS University Islamabad, Wah Campus},
  year={2025},
  month={July},
  type={Bachelor of Science in Computer Science Thesis},
  note={Supervised by Dr. Saeed-Ur-Rehman}
}
```

---

## 09. LICENSE & ATTRIBUTION

Licensed under the Apache License, Version 2.0 (the "License").
You may obtain a copy of the License at:

    http://www.apache.org/licenses/LICENSE-2.0

Author: Baseer Hassan (baseerhassan512@gmail.com)  
Department of Computer Science, COMSATS University Islamabad
