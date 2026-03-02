"""Baseline training: YOLOv8 with P2 head (SRFD) on head detection datasets.

Paper: "Using Cross-Domain Detection Loss to Infer Multi-Scale Information
for Improved Tiny Head Tracking" (FG2025, arXiv 2505.22677).

This script trains the Student or Teacher baseline models (Table 1 & 2).

Usage:
    # Student baseline (YOLOv8s-p2) on CrowdHuman
    python tools/train_baseline.py --model yolov8s-p2 --data tools/configs/CrowdHuman.yaml

    # Teacher baseline (YOLOv8l-p2) on CrowdHuman
    python tools/train_baseline.py --model yolov8l-p2 --data tools/configs/CrowdHuman.yaml

    # Student baseline on HT21
    python tools/train_baseline.py --model yolov8s-p2 --data tools/configs/mot_ht.yaml
"""

import argparse

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description='Baseline YOLOv8-P2 Training')
    parser.add_argument('--model', type=str, default='yolov8s-p2',
                        help='Model config name (e.g., yolov8n-p2, yolov8s-p2, yolov8l-p2)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML (e.g., tools/configs/CrowdHuman.yaml)')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default='runs/baseline',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='',
                        help='Experiment name (default: auto-generated from model)')
    args = parser.parse_args()

    # Create model from YAML (ultralytics auto-resolves config names)
    model = YOLO(f'{args.model}.yaml')

    # Train
    exp_name = args.name or args.model
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=exp_name,
    )

    # Validate
    model.val()


if __name__ == '__main__':
    main()
