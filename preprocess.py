"""
Research-grade Deepfake Preprocessing Pipeline.

Features:
  - 30-frame uniform sampling per video
  - SCRFD face detector + IoU bounding-box tracking
  - Bounding-box filling mechanism for missing segments (up to 5 consecutive frames)
  - Sharpness quality control via Laplacian variance
  - 0.4 crop margin expansion + 224x224 linear interpolation resize
  - Lock-free sequential execution (prevents OpenCV CPU thrashing)
"""

import os
import cv2
cv2.setNumThreads(0)  # Prevent OpenCV CPU context switching thrashing

import json
import logging
import argparse
import random
import shutil
import uuid
import gc
import numpy as np
import pandas as pd
import onnxruntime as ort
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from tqdm import tqdm
import insightface

import config


def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    cv2.setRNGSeed(seed)


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


@dataclass
class VideoMeta:
    video_id:   str
    video_path: str
    label:      str
    split:      str
    dataset_id: str = "unknown"


@dataclass
class FrameRecord:
    frame_idx:       int
    sample_position: int
    detected:        bool
    filled:          bool
    fill_source:     Optional[int]
    det_score:       Optional[float]
    bbox:            Optional[list]
    blur_var:        Optional[float]
    align_method:    str


@dataclass
class VideoAuditLog:
    video_id:            str
    dataset_id:          str
    status:              str
    reject_reason:       Optional[str]
    total_frames:        int
    sampled_indices:     list
    missing_count:       int
    max_consecutive_gap: int
    frame_records:       list = field(default_factory=list)


def get_dataset_splits(split_file_path: str, logger: logging.Logger) -> dict:
    splits_dict = {"train": [], "val": [], "test": []}
    logger.info(f"Loading split CSV file: {split_file_path}")
    df = pd.read_csv(split_file_path)

    skipped = 0
    for _, row in df.iterrows():
        video_path = row.get("video_path")
        label      = row.get("label")
        split_name = row.get("split")

        if pd.isna(video_path) or pd.isna(label) or pd.isna(split_name):
            skipped += 1
            continue

        video_path = str(video_path).strip()
        label      = str(label).lower().strip()
        split_name = str(split_name).lower().strip()

        if label not in ("real", "fake") or split_name not in splits_dict or not os.path.exists(video_path):
            skipped += 1
            continue

        filename = row.get("video_filename") or row.get("filename")
        video_id = row.get("video_id")
        if pd.isna(video_id):
            video_id = Path(str(filename) if filename else video_path).stem

        splits_dict[split_name].append(VideoMeta(
            video_id   = str(video_id),
            video_path = video_path,
            label      = label,
            split      = split_name,
            dataset_id = str(row.get("dataset_id", "dataset")),
        ))

    logger.info(f"Splits loaded successfully. Skipped {skipped} invalid rows.")
    return splits_dict


def sample_frame_indices(total_frames: int, num_frames: int, min_total_frames: int) -> Optional[list]:
    if total_frames < min_total_frames:
        return None
    indices = np.round(np.linspace(0, total_frames - 1, num_frames)).astype(int)
    unique_indices = sorted(list(set(int(x) for x in indices)))
    return unique_indices if len(unique_indices) >= num_frames else None


def extract_frames_sequential(video_path: str, target_indices: list) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}

    result = {}
    target_set = set(target_indices)
    max_idx = max(target_indices) if target_indices else -1

    current_idx = 0
    while current_idx <= max_idx:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if current_idx in target_set:
            result[current_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_idx += 1

    cap.release()
    return result


class DetectionOnlyRetinaFace:
    def __init__(self, model_path: str, det_size=(640, 640), min_score=0.5, min_face_px=30, ctx_id=0):
        self.detector = insightface.model_zoo.get_model(
            model_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.detector.prepare(ctx_id=ctx_id, input_size=det_size)
        self.min_score = min_score
        self.min_face_px = min_face_px

    def detect_batch(self, rgb_frames):
        results = []
        for frame in rgb_frames:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            bboxes, _ = self.detector.detect(bgr_frame)

            frame_dets = []
            if bboxes is not None:
                for i in range(bboxes.shape[0]):
                    bbox = bboxes[i, 0:4]
                    score = float(bboxes[i, 4])
                    if score < self.min_score:
                        continue
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if w >= self.min_face_px and h >= self.min_face_px:
                        frame_dets.append({"bbox": bbox, "score": score, "area": float(w * h)})
            results.append(frame_dets)
        return results


def find_det10g_model() -> str:
    candidates = [
        os.path.expanduser("~/.insightface/models/buffalo_l/det_10g.onnx"),
        "/root/.insightface/models/buffalo_l/det_10g.onnx",
        "/kaggle/working/buffalo_l/det_10g.onnx",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))
        del app
    except Exception:
        pass

    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("SCRFD det_10g.onnx model weight file not found.")


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    union = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]) + (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


class IoUTracker:
    def __init__(self, iou_threshold=0.3, scene_cut_iou=0.1):
        self.iou_threshold = iou_threshold
        self.scene_cut_iou = scene_cut_iou
        self.current_bbox  = None

    def update(self, detections: list) -> Optional[dict]:
        if not detections:
            return None
        if self.current_bbox is None:
            best = max(detections, key=lambda d: d["area"])
            self.current_bbox = best["bbox"].copy()
            return best

        ious = [compute_iou(self.current_bbox, d["bbox"]) for d in detections]
        best_idx = int(np.argmax(ious))

        if ious[best_idx] < self.scene_cut_iou:
            best = max(detections, key=lambda d: d["area"])
            self.current_bbox = best["bbox"].copy()
            return best

        selected = detections[best_idx]
        self.current_bbox = selected["bbox"].copy()
        return selected


def extract_bbox_crop(rgb_frame, bbox, margin, target_size):
    h, w = rgb_frame.shape[:2]
    x1, y1, x2, y2 = bbox
    expand = max(x2 - x1, y2 - y1) * margin
    rx1, ry1 = max(0, x1 - expand / 2), max(0, y1 - expand / 2)
    rx2, ry2 = min(w, x2 + expand / 2), min(h, y2 + expand / 2)
    crop = rgb_frame[int(np.floor(ry1)):int(np.ceil(ry2)), int(np.floor(rx1)):int(np.ceil(rx2))]
    if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
        return None
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LINEAR)


def compute_blur_variance(rgb_frame: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def save_audit_log(audit: VideoAuditLog, output_root: str, split: str) -> None:
    log_dir = os.path.join(output_root, "logs", split)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{audit.video_id}.json")
    with open(log_path, "w") as f:
        json.dump(asdict(audit), f, indent=2)


def run_pipeline(
    split_file_path:  str,
    output_root_dir:  str,
    num_frames:       int   = config.NUM_FRAMES,
    target_size:      int   = config.FRAME_SIZE,
    crop_margin:      float = config.DEFAULT_CROP_MARGIN,
    min_face_score:   float = config.DEFAULT_MIN_FACE_SCORE,
    min_face_pixels:  int   = config.DEFAULT_MIN_FACE_PIXELS,
    min_total_frames: int   = config.DEFAULT_MIN_TOTAL_FRAMES,
    max_missing_frac: float = config.DEFAULT_MAX_MISSING_FRAC,
    max_consec_gap:   int   = config.DEFAULT_MAX_CONSEC_GAP,
    ctx_id:           int   = 0,
    seed:             int   = config.RANDOM_SEED,
):
    set_global_seeds(seed)
    os.makedirs(output_root_dir, exist_ok=True)
    logger = get_logger("deepfake_pipeline", log_file=os.path.join(output_root_dir, "pipeline.log"))

    logger.info("Initializing Preprocessing Pipeline...")
    model_path = find_det10g_model()
    detector = DetectionOnlyRetinaFace(model_path, min_score=min_face_score, min_face_px=min_face_pixels, ctx_id=ctx_id)
    dataset_splits = get_dataset_splits(split_file_path, logger)

    stats = {"saved": 0, "rejected": 0, "errors": 0}

    for split_name in ["train", "val", "test"]:
        video_list = dataset_splits.get(split_name, [])
        if not video_list:
            continue

        logger.info(f"Processing [{split_name.upper()}] split — {len(video_list)} total videos")

        for meta in tqdm(video_list, desc=split_name):
            frame_dict = {}
            ordered_frames = []
            valid_frames = []
            det_results_valid = []
            crops = []

            tmp_video_path = os.path.join(os.environ.get("TEMP", "/tmp"), f"{uuid.uuid4().hex}.mp4")

            try:
                shutil.copy(meta.video_path, tmp_video_path)

                cap = cv2.VideoCapture(tmp_video_path)
                if not cap.isOpened():
                    stats["rejected"] += 1
                    continue
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                sampled_indices = sample_frame_indices(total_frames, num_frames, min_total_frames)
                if not sampled_indices:
                    stats["rejected"] += 1
                    continue

                audit = VideoAuditLog(
                    video_id=meta.video_id, dataset_id=meta.dataset_id,
                    status="rejected", reject_reason=None, total_frames=total_frames,
                    sampled_indices=sampled_indices, missing_count=0, max_consecutive_gap=0
                )

                frame_dict = extract_frames_sequential(tmp_video_path, sampled_indices)
                ordered_frames = [frame_dict.get(idx) for idx in sampled_indices]
                valid_frames = [f for f in ordered_frames if f is not None]

                if not valid_frames or len(valid_frames) < num_frames:
                    audit.reject_reason = "frame_extraction_failed"
                    save_audit_log(audit, output_root_dir, split_name)
                    stats["rejected"] += 1
                    continue

                det_results_valid = detector.detect_batch(valid_frames)

                valid_iter = iter(det_results_valid)
                det_per_pos = [next(valid_iter) if f is not None else None for f in ordered_frames]

                tracker = IoUTracker()
                tracked = [tracker.update(d) if d else None for d in det_per_pos]
                valid_positions = [i for i, t in enumerate(tracked) if t is not None]
                missing_count = num_frames - len(valid_positions)

                max_gap, current_gap = 0, 0
                for t in tracked:
                    if t is None:
                        current_gap += 1
                        max_gap = max(max_gap, current_gap)
                    else:
                        current_gap = 0

                audit.missing_count = missing_count
                audit.max_consecutive_gap = max_gap

                if missing_count / num_frames > max_missing_frac:
                    audit.reject_reason = f"too_many_missing:{missing_count}/{num_frames}"
                    save_audit_log(audit, output_root_dir, split_name)
                    stats["rejected"] += 1
                    continue

                if max_gap > max_consec_gap:
                    audit.reject_reason = f"consecutive_gap_exceeded:{max_gap}"
                    save_audit_log(audit, output_root_dir, split_name)
                    stats["rejected"] += 1
                    continue

                filled_results = []
                for pos in range(num_frames):
                    if tracked[pos] is not None:
                        filled_results.append({"detection": tracked[pos], "filled": False, "fill_source": None})
                    else:
                        prev_v = max((p for p in valid_positions if p < pos), default=None)
                        next_v = min((p for p in valid_positions if p > pos), default=None)
                        source = prev_v if prev_v is not None else next_v
                        filled_results.append({"detection": tracked[source], "filled": True, "fill_source": source})

                save_dir = os.path.join(output_root_dir, meta.split, meta.label, meta.video_id)
                os.makedirs(save_dir, exist_ok=True)

                success_flag = True
                frame_records = []

                for pos, (frame_idx, entry) in enumerate(zip(sampled_indices, filled_results)):
                    rgb_frame = frame_dict.get(frame_idx)
                    if rgb_frame is None:
                        rgb_frame = valid_frames[0]

                    crop = extract_bbox_crop(rgb_frame, entry["detection"]["bbox"], crop_margin, target_size)
                    if crop is None:
                        success_flag = False
                        break

                    blur_var = compute_blur_variance(crop)
                    crops.append(crop)

                    frame_records.append(FrameRecord(
                        frame_idx=frame_idx, sample_position=pos,
                        detected=not entry["filled"], filled=entry["filled"],
                        fill_source=int(entry["fill_source"]) if entry["fill_source"] is not None else None,
                        det_score=float(entry["detection"]["score"]),
                        bbox=[float(x) for x in entry["detection"]["bbox"]],
                        blur_var=blur_var,
                        align_method="filled" if entry["filled"] else "detection"
                    ))

                    bgr_crop = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(save_dir, f"{pos:04d}.png"), bgr_crop, [cv2.IMWRITE_PNG_COMPRESSION, 1])

                if success_flag:
                    audit.frame_records = [asdict(r) for r in frame_records]
                    audit.status = "ok"
                    save_audit_log(audit, output_root_dir, split_name)
                    stats["saved"] += 1
                else:
                    audit.reject_reason = "degenerate_crop_encountered"
                    save_audit_log(audit, output_root_dir, split_name)
                    stats["rejected"] += 1

            except Exception as e:
                logger.error(f"Error processing {meta.video_id}: {e}")
                stats["errors"] += 1

            finally:
                if os.path.exists(tmp_video_path):
                    try:
                        os.remove(tmp_video_path)
                    except Exception:
                        pass
                del frame_dict, ordered_frames, valid_frames, det_results_valid, crops
                gc.collect()

    logger.info(f"Pipeline Execution Finished — Saved: {stats['saved']} | Rejected: {stats['rejected']} | Errors: {stats['errors']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Video Preprocessing Pipeline")
    parser.add_argument("--split_file", type=str, required=True, help="Path to split CSV file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save preprocessed dataset")
    parser.add_argument("--num_frames", type=int, default=config.NUM_FRAMES, help="Number of frames per video")
    parser.add_argument("--target_size", type=int, default=config.FRAME_SIZE, help="Output crop image resolution")
    parser.add_argument("--crop_margin", type=float, default=config.DEFAULT_CROP_MARGIN, help="Face crop margin expansion")
    parser.add_argument("--min_score", type=float, default=config.DEFAULT_MIN_FACE_SCORE, help="Minimum face detection score")
    parser.add_argument("--ctx_id", type=int, default=0, help="GPU device ID (-1 for CPU)")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED, help="Random seed for reproducibility")

    args = parser.parse_args()

    run_pipeline(
        split_file_path=args.split_file,
        output_root_dir=args.output_dir,
        num_frames=args.num_frames,
        target_size=args.target_size,
        crop_margin=args.crop_margin,
        min_face_score=args.min_score,
        ctx_id=args.ctx_id,
        seed=args.seed,
    )
