"""Extract and align faces from DFDC video files."""

import json
import cv2
import numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
import torch
from tqdm import tqdm

from src.config import (
    RAW_DATA_DIR, FACES_DIR, FACE_SIZE, FACE_MARGIN,
    FRAMES_PER_VIDEO, MAX_VIDEOS_PER_PART, MIN_FACE_CONFIDENCE
)


def create_mtcnn(device="cuda"):
    return MTCNN(
        image_size=FACE_SIZE,
        margin=int(FACE_SIZE * FACE_MARGIN),
        min_face_size=60,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        device=device,
    )


def sample_frames(video_path, num_frames):
    """Sample evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((idx, frame_rgb))

    cap.release()
    return frames


def extract_faces_from_video(video_path, mtcnn, output_dir, label):
    """Extract faces from a single video, save as individual images."""
    frames = sample_frames(video_path, FRAMES_PER_VIDEO)
    if not frames:
        return 0

    video_name = video_path.stem
    video_output_dir = output_dir / label / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for frame_idx, frame in frames:
        boxes, probs = mtcnn.detect(frame)
        if boxes is None:
            continue

        for i, (box, prob) in enumerate(zip(boxes, probs)):
            if prob < MIN_FACE_CONFIDENCE:
                continue

            x1, y1, x2, y2 = [int(b) for b in box]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            face_resized = cv2.resize(face, (FACE_SIZE, FACE_SIZE))
            filename = f"frame{frame_idx:04d}_face{i}.jpg"
            cv2.imwrite(
                str(video_output_dir / filename),
                cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95]
            )
            saved_count += 1

    return saved_count


def extract_all_faces(parts=None):
    """Extract faces from all DFDC parts. parts is a list like [0, 1, 2, 3]."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = create_mtcnn(device)
    FACES_DIR.mkdir(parents=True, exist_ok=True)

    if parts is None:
        part_dirs = sorted(RAW_DATA_DIR.glob("part_*"))
    else:
        part_dirs = [RAW_DATA_DIR / f"part_{p:02d}" for p in parts]

    total_faces = 0
    for part_dir in part_dirs:
        if not part_dir.exists():
            print(f"Skipping {part_dir} — not found")
            continue

        metadata_file = part_dir / "metadata.json"
        if not metadata_file.exists():
            print(f"No metadata.json in {part_dir}, skipping")
            continue

        with open(metadata_file) as f:
            metadata = json.load(f)

        videos = list(part_dir.glob("*.mp4"))
        if MAX_VIDEOS_PER_PART:
            videos = videos[:MAX_VIDEOS_PER_PART]

        print(f"\nProcessing {part_dir.name}: {len(videos)} videos")

        for video_path in tqdm(videos, desc=part_dir.name):
            video_name = video_path.name
            if video_name not in metadata:
                continue

            label = metadata[video_name]["label"].lower()  # "REAL" or "FAKE"
            n_faces = extract_faces_from_video(video_path, mtcnn, FACES_DIR, label)
            total_faces += n_faces

    print(f"\nTotal faces extracted: {total_faces}")
    print(f"  Real: {sum(1 for _ in (FACES_DIR / 'real').rglob('*.jpg')) if (FACES_DIR / 'real').exists() else 0}")
    print(f"  Fake: {sum(1 for _ in (FACES_DIR / 'fake').rglob('*.jpg')) if (FACES_DIR / 'fake').exists() else 0}")

    return total_faces


if __name__ == "__main__":
    extract_all_faces(parts=[0, 1, 2, 3])
