"""ByteTrack MOT inference on HT21 sequences.

Paper: "Using Cross-Domain Detection Loss to Infer Multi-Scale Information
for Improved Tiny Head Tracking" (FG2025, arXiv 2505.22677).

Runs YOLO detection + ByteTrack tracking on HT21 sequences and outputs
results in MOTChallenge format for evaluation with TrackEval.

Usage:
    python tools/run_tracking.py \
        --weights runs/cddl/yolov8s-p2_CDDL/weights/best.pt \
        --tracker_name CDDL_YOLOv8s \
        --seq_dir /path/to/ht21_sequences \
        --output_dir /path/to/trackeval/data/trackers/mot_challenge/HT21-train
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


# HT21 sequence information
SEQUENCES = {
    "HT21-01": 429,
    "HT21-02": 3315,
    "HT21-03": 1000,
    "HT21-04": 997,
}


def run_tracking(weights, tracker_name, seq_dir, output_dir, imgsz=1920, conf=0.1):
    """Run tracking on each HT21 sequence and save in MOTChallenge format."""
    model = YOLO(weights)
    seq_dir = Path(seq_dir)
    output_dir = Path(output_dir)

    # Create output directory
    out_dir = output_dir / tracker_name / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq_name, seq_len in SEQUENCES.items():
        seq_path = seq_dir / seq_name
        if not seq_path.exists():
            print(f"[SKIP] {seq_name}: directory not found ({seq_path})")
            continue

        print(f"\n{'='*60}")
        print(f"[TRACK] {seq_name} ({seq_len} frames) - imgsz={imgsz}, conf={conf}")
        print(f"{'='*60}")

        # YOLO tracking (persist=True maintains tracks, stream=True saves RAM)
        results = model.track(
            source=str(seq_path),
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=imgsz,
            conf=conf,
            verbose=False,
            stream=True,
        )

        # Save in MOTChallenge format
        out_file = out_dir / f"{seq_name}.txt"
        lines = []
        for frame_idx, result in enumerate(results):
            frame_id = frame_idx + 1  # 1-based
            if result.boxes is None or result.boxes.id is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                track_id = int(boxes.id[i].item())
                # xyxy -> xywh conversion
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                w = x2 - x1
                h = y2 - y1
                conf_score = boxes.conf[i].item()
                lines.append(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},{conf_score:.6f},-1,-1,-1")

        with open(out_file, "w") as f:
            f.write("\n".join(lines) + "\n" if lines else "")

        # Verify: max frame_id should match sequence length
        if lines:
            max_frame = max(int(line.split(",")[0]) for line in lines)
            n_tracks = len(set(int(line.split(",")[1]) for line in lines))
            print(f"[DONE] {seq_name}: {len(lines)} detections, {n_tracks} tracks, "
                  f"max_frame={max_frame}/{seq_len}")
            if max_frame != seq_len:
                print(f"[WARN] max_frame ({max_frame}) != seqLength ({seq_len})")
        else:
            print(f"[WARN] {seq_name}: no detections")

        print(f"[SAVE] {out_file}")

    print(f"\n{'='*60}")
    print(f"[COMPLETE] Tracker: {tracker_name}")
    print(f"[OUTPUT]   {out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HT21 ByteTrack MOT Inference")
    parser.add_argument("--weights", type=str, required=True, help="YOLO model weights path")
    parser.add_argument("--tracker_name", type=str, required=True, help="Tracker name for output directory")
    parser.add_argument("--seq_dir", type=str, required=True, help="Directory containing HT21 sequence folders")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="TrackEval base output directory (e.g., .../HT21-train)")
    parser.add_argument("--imgsz", type=int, default=1920, help="Inference image size (default: 1920)")
    parser.add_argument("--conf", type=float, default=0.1, help="Detection confidence threshold (default: 0.1)")
    args = parser.parse_args()

    run_tracking(args.weights, args.tracker_name, args.seq_dir, args.output_dir, args.imgsz, args.conf)
