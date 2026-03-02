# CDDL: Cross-Domain Detection Loss for Tiny Head Tracking

Official implementation of **"Using Cross-Domain Detection Loss to Infer Multi-Scale Information for Improved Tiny Head Tracking"** (IEEE FG 2025).

[[arXiv](https://arxiv.org/abs/2505.22677)]

## Overview

We propose **Cross-Domain Detection Loss (CDDL)**, a knowledge distillation framework that transfers detection knowledge from a large teacher model to a compact student model for tiny head tracking. Our approach combines:

- **SRFD (Small Resolution Feature Detection)**: YOLOv8 with a P2 detection head for detecting small objects at higher resolution feature maps (stride 4).
- **CDDL**: A three-component KD loss (CIoU_KD + DFL_KD + BCE_KD) where the student mimics the teacher's detection outputs.
- **MSI (Multi-Scale Input)**: A learnable multi-resolution fusion module that provides multi-scale information to the student.

## Installation

```bash
git clone https://github.com/wltnkim/CDDL-TinyHeadTracking.git
cd CDDL-TinyHeadTracking
pip install -e .
```

### Requirements
- Python >= 3.8
- PyTorch >= 1.8.0
- See `requirements.txt` for full dependencies

## Dataset Preparation

### CrowdHuman
1. Download from [CrowdHuman](http://www.crowdhuman.org/)
2. Convert annotations to YOLO format (1 class: `head`)
3. Update `tools/configs/CrowdHuman.yaml` with your dataset path

### HT21 (Head Tracking 21)
1. Download from [MOTChallenge](https://motchallenge.net/data/Head_Tracking_21/)
2. Convert annotations to YOLO format
3. Update `tools/configs/mot_ht.yaml` with your dataset path

## Training

### Step 1: Train Teacher Baseline (YOLOv8l-p2)

```bash
python tools/train_baseline.py \
    --model yolov8l-p2 \
    --data tools/configs/CrowdHuman.yaml \
    --epochs 300 --batch 8 --imgsz 640
```

### Step 2: Train Student Baseline (YOLOv8s-p2)

```bash
python tools/train_baseline.py \
    --model yolov8s-p2 \
    --data tools/configs/CrowdHuman.yaml \
    --epochs 300 --batch 8 --imgsz 640
```

### Step 3: CDDL Training (Student + Teacher KD)

```bash
# CDDL only
python tools/train_cddl.py \
    --teacher_weights runs/baseline/yolov8l-p2/weights/best.pt \
    --data tools/configs/CrowdHuman.yaml \
    --epochs 300 --batch 8 --imgsz 640

# CDDL + Multi-Scale Input (CDDL_MS)
python tools/train_cddl.py \
    --teacher_weights runs/baseline/yolov8l-p2/weights/best.pt \
    --data tools/configs/CrowdHuman.yaml \
    --multi_scale \
    --epochs 300 --batch 8 --imgsz 640
```

## Tracking Evaluation

Run ByteTrack tracking on HT21 sequences:

```bash
python tools/run_tracking.py \
    --weights runs/cddl/yolov8s-p2_CDDL/weights/best.pt \
    --tracker_name CDDL_YOLOv8s \
    --seq_dir /path/to/HT21/sequences \
    --output_dir /path/to/TrackEval/data/trackers/mot_challenge/HT21-train \
    --imgsz 1920 --conf 0.1
```

Then evaluate with [TrackEval](https://github.com/JonathonLuiten/TrackEval):

```bash
python scripts/run_mot_challenge.py \
    --BENCHMARK HT21 --SPLIT_TO_EVAL train \
    --TRACKERS_TO_EVAL CDDL_YOLOv8s \
    --METRICS HOTA CLEAR Identity
```

## Results

### CrowdHuman Detection (Table II)

| Method | Resolution | mAP50 | mAP50-95 | FLOPs |
|--------|-----------|-------|----------|-------|
| YOLOv8n | 640 | 0.608 | 0.361 | 4.10B |
| YOLOv8s | 640 | 0.615 | 0.372 | 14.32B |
| YOLOv8l | 640 | 0.642 | 0.395 | 82.70B |
| YOLOv8n + SRFD + CDDL + MS | multi | 0.718 | 0.452 | 6.18B |
| YOLOv8s + SRFD + CDDL + MS | multi | 0.735 | 0.470 | 18.47B |
| YOLOv8m + SRFD + CDDL + MS | multi | 0.740 | 0.472 | 49.24B |
| YOLOv8l + SRFD + MS | multi | **0.742** | **0.474** | 102.80B |

### CroHD Tracking (Table I)

| Tracker | MOTA | Time/Frame | FLOPs | MOTA/BFLOPs |
|---------|------|-----------|-------|-------------|
| YOLOv8s | 42.64 | 26.41ms | 14.32B | 2.977 |
| YOLOv8l | 44.72 | 30.46ms | 82.70B | 0.540 |
| YOLOv8s + SRFD + CDDL | 46.60 | 27.51ms | 18.47B | 2.523 |
| YOLOv8s + SRFD + CDDL + MS | **56.22** | 27.52ms | 18.47B | **3.043** |
| YOLOv8l + SRFD | 47.00 | 32.78ms | 102.80B | 0.457 |
| YOLOv8l + SRFD + MS | 57.13 | 32.76ms | 102.80B | 0.555 |

## Project Structure

```
CDDL-TinyHeadTracking/
├── README.md
├── LICENSE                              # AGPL-3.0
├── requirements.txt
├── setup.py
├── ultralytics/                         # Ultralytics v8.0.223 + CDDL modifications
│   ├── cfg/
│   │   ├── default.yaml                 # +cddl_* parameters
│   │   ├── __init__.py                  # +cddl_alpha in CFG_FLOAT_KEYS
│   │   └── models/v8/
│   │       ├── yolov8.yaml
│   │       └── yolov8-p2.yaml           # SRFD (P2 detection head)
│   ├── nn/
│   │   ├── tasks.py                     # +CDDLDetectionModel
│   │   └── modules/
│   │       └── block.py                 # +MultiScaleInput class
│   ├── utils/
│   │   └── loss.py                      # +CDDLBboxLoss, CDDLLoss
│   └── models/yolo/detect/
│       └── train_cddl.py               # CDDLTrainer
└── tools/
    ├── train_baseline.py                # Baseline training script
    ├── train_cddl.py                    # CDDL training script
    ├── run_tracking.py                  # ByteTrack tracking + MOT evaluation
    └── configs/
        ├── CrowdHuman.yaml              # Dataset config
        └── mot_ht.yaml                  # Dataset config
```

## Citation

```bibtex
@inproceedings{kim2025cddl,
  title={Using Cross-Domain Detection Loss to Infer Multi-Scale Information for Improved Tiny Head Tracking},
  author={Kim, Jisu and Mattingly, Alex and Lee, Eung-Joo and Riggan, Benjamin S.},
  booktitle={IEEE International Conference on Automatic Face and Gesture Recognition (FG)},
  year={2025}
}
```

## License

This project is released under the [AGPL-3.0 License](LICENSE), inherited from the [Ultralytics](https://github.com/ultralytics/ultralytics) codebase (v8.0.223).

## Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for the base detection framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) for the tracking algorithm
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) for MOT evaluation metrics
