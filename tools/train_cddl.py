"""CDDL Training: Student + Teacher knowledge distillation.

Paper: "Using Cross-Domain Detection Loss to Infer Multi-Scale Information
for Improved Tiny Head Tracking" (FG2025, arXiv 2505.22677).

This script trains the student model with CDDL (Table 1 & 2).
Requires a pre-trained teacher model (.pt file).

Usage:
    # CDDL only (without Multi-Scale Input)
    python tools/train_cddl.py \
        --teacher_weights /path/to/yolov8l-p2-best.pt \
        --data tools/configs/CrowdHuman.yaml

    # CDDL + Multi-Scale Input (CDDL_MS)
    python tools/train_cddl.py \
        --teacher_weights /path/to/yolov8l-p2-best.pt \
        --data tools/configs/CrowdHuman.yaml \
        --multi_scale

    # CDDL on HT21
    python tools/train_cddl.py \
        --teacher_weights /path/to/yolov8l-p2-best.pt \
        --data tools/configs/mot_ht.yaml
"""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.models.yolo.detect.train_cddl import CDDLTrainer


def main():
    parser = argparse.ArgumentParser(description='CDDL Training')
    parser.add_argument('--student_cfg', type=str, default='yolov8s-p2',
                        help='Student model config name (e.g., yolov8n-p2, yolov8s-p2)')
    parser.add_argument('--teacher_weights', type=str, required=True,
                        help='Path to teacher .pt weights (e.g., yolov8l-p2 best.pt)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to dataset YAML (e.g., tools/configs/CrowdHuman.yaml)')
    parser.add_argument('--multi_scale', action='store_true',
                        help='Enable Multi-Scale Input module')
    parser.add_argument('--cddl_alpha', type=float, default=1.0,
                        help='KD loss weight: L = L_det + cddl_alpha * L_KD')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--project', type=str, default='runs/cddl',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='',
                        help='Experiment name (default: auto-generated)')
    args = parser.parse_args()

    # Student model from YAML
    model = YOLO(f'{args.student_cfg}.yaml')

    # Auto-generate experiment name
    ms_suffix = '_MS' if args.multi_scale else ''
    exp_name = args.name or f'{args.student_cfg}_CDDL{ms_suffix}'

    # CDDL training
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        trainer=CDDLTrainer,
        cddl_teacher_weights=args.teacher_weights,
        cddl_multi_scale=args.multi_scale,
        cddl_alpha=args.cddl_alpha,
        project=args.project,
        name=exp_name,
    )

    # Validate
    model.val()


if __name__ == '__main__':
    main()
