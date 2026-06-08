"""Drone detection inference on video files using YOLOv12.

Runs locally (CPU/GPU) or in Colab after downloading weights from Drive.

Usage:
    python scripts/video_inference.py \
        --weights path/to/yolo12s_drone_best.pt \
        --source path/to/video.mp4 \
        --output path/to/output.mp4 \
        --conf 0.35

    # Webcam / RTSP stream:
    python scripts/video_inference.py \
        --weights weights/yolo12s_drone_best.pt \
        --source 0 \          # webcam index
        --conf 0.35 --show
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = ['DRONE', 'AIRPLANE', 'HELICOPTER', 'BIRD']

CLASS_BGR = {
    'DRONE':      (0,   0,   255),
    'AIRPLANE':   (255, 0,   0  ),
    'HELICOPTER': (0,   255, 0  ),
    'BIRD':       (0,   255, 255),
}


def draw_detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    fps: float,
) -> np.ndarray:
    """Draw bounding boxes and FPS counter on a frame.

    Args:
        frame: BGR image as numpy array.
        boxes: Array of [x1, y1, x2, y2] bounding boxes.
        scores: Confidence scores for each detection.
        class_ids: Integer class indices for each detection.
        fps: Current inference FPS.

    Returns:
        Annotated frame.
    """
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for box, score, cls_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f'cls{cls_id}'
        color    = CLASS_BGR.get(cls_name, (255, 255, 255))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f'{cls_name} {score:.2f}'
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated,
                      (x1, y1 - th - baseline - 4),
                      (x1 + tw + 4, y1),
                      color, -1)
        cv2.putText(annotated, label,
                    (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

    # FPS counter (bottom-left)
    fps_text = f'FPS: {fps:.1f}'
    cv2.putText(annotated, fps_text,
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 255, 255), 2, cv2.LINE_AA)

    return annotated


def run_inference(
    weights: str | Path,
    source: str | int,
    output: str | Path | None,
    conf: float,
    iou: float,
    max_frames: int,
    show: bool,
    device: str,
) -> None:
    """Run YOLO inference on video source.

    Args:
        weights: Path to YOLO .pt weights file.
        source: Video file path or webcam index.
        output: Output video path. None to skip saving.
        conf: Confidence threshold.
        iou: IoU NMS threshold.
        max_frames: Max frames to process (0 = unlimited).
        show: Whether to display frames in a window.
        device: Device string ('cpu', '0', '1', etc.).
    """
    model = YOLO(str(weights))
    print(f'Model loaded: {weights}')

    cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not cap.isOpened():
        raise ValueError(f'Cannot open source: {source}')

    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f'Source: {source}')
    print(f'Resolution: {frame_w}x{frame_h} @ {orig_fps:.1f} FPS')
    if total > 0:
        print(f'Total frames: {total}')

    writer = None
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output), fourcc, orig_fps, (frame_w, frame_h))
        print(f'Output: {output}')

    det_counts: dict[str, int] = {n: 0 for n in CLASS_NAMES}
    frame_times: list[float] = []
    frame_idx = 0

    try:
        while True:
            if max_frames > 0 and frame_idx >= max_frames:
                break

            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            device_id = int(device) if device.isdigit() else device
            results = model.predict(frame, conf=conf, iou=iou,
                                    device=device_id, verbose=False)
            t1 = time.perf_counter()
            frame_times.append(t1 - t0)

            result = results[0]
            boxes     = result.boxes.xyxy.cpu().numpy()  if len(result.boxes) else np.empty((0, 4))
            scores    = result.boxes.conf.cpu().numpy()  if len(result.boxes) else np.empty(0)
            class_ids = result.boxes.cls.cpu().numpy().astype(int) if len(result.boxes) else np.empty(0, dtype=int)

            for cls_id in class_ids:
                if cls_id < len(CLASS_NAMES):
                    det_counts[CLASS_NAMES[cls_id]] += 1

            avg_fps = 1.0 / np.mean(frame_times[-15:])
            annotated = draw_detections(frame, boxes, scores, class_ids, avg_fps)

            if writer is not None:
                writer.write(annotated)

            if show:
                cv2.imshow('Drone Detection', annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f'  Frame {frame_idx} | FPS {avg_fps:.1f}')

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    avg_fps_total = 1.0 / np.mean(frame_times) if frame_times else 0
    print(f'\nDone! Processed {frame_idx} frames')
    print(f'Average FPS: {avg_fps_total:.1f}')
    print(f'Detections by class: {det_counts}')
    if output and Path(output).exists():
        size_mb = Path(output).stat().st_size / 1e6
        print(f'Output file: {output} ({size_mb:.1f} MB)')


def main() -> None:
    parser = argparse.ArgumentParser(description='Drone detection video inference')
    parser.add_argument('--weights',    required=True, help='Path to YOLO .pt weights')
    parser.add_argument('--source',     required=True, help='Video path or webcam index (0, 1, ...)')
    parser.add_argument('--output',     default=None,  help='Output video path (optional)')
    parser.add_argument('--conf',       type=float, default=0.35, help='Confidence threshold')
    parser.add_argument('--iou',        type=float, default=0.45, help='IoU NMS threshold')
    parser.add_argument('--max_frames', type=int,   default=0,    help='Max frames (0=all)')
    parser.add_argument('--show',       action='store_true',      help='Display frames')
    parser.add_argument('--device',     default='0',              help='Device: cpu or GPU index')
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source

    run_inference(
        weights    = args.weights,
        source     = source,
        output     = args.output,
        conf       = args.conf,
        iou        = args.iou,
        max_frames = args.max_frames,
        show       = args.show,
        device     = args.device,
    )


if __name__ == '__main__':
    main()
