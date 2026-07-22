"""Resolve videos to holistic poses.

Two sources:
  * Pre-computed corpus poses under ``TRANSFORMED_VIDEOS_DIR`` keyed by the
    md5 of the source mp4 (fast, no extraction).
  * On-the-fly MediaPipe holistic extraction for videos that were never
    ingested (e.g. DGS Types dictionary clips), cached to disk.

Both paths yield the identical 586-point, 50 fps MediaPipe holistic layout,
so poses from either source are directly comparable.
"""

from __future__ import annotations

import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pose_format import Pose

from fluent_pose_synthesis.stitching.config import POSE_FPS, TRANSFORMED_VIDEOS_DIR


def video_md5(video_path: Path, chunk_size: int = 1 << 20) -> str:
    """md5 hex digest of a file, streamed so large mp4s stay off the heap."""
    h = hashlib.md5()
    with open(video_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def transformed_pose_path(md5: str) -> Path:
    return TRANSFORMED_VIDEOS_DIR / md5 / "holistic.pose"


def load_pose(path: Path) -> Pose:
    with open(path, "rb") as f:
        return Pose.read(f.read())


@lru_cache(maxsize=8)
def load_document_pose(video_path: str) -> Pose:
    """Full-document holistic pose for a corpus video (md5 lookup).

    LRU-cached because a single document pose (~300 MB) is sliced into many
    sentences; we never want to reload it per sentence.
    """
    md5 = video_md5(Path(video_path))
    pose_path = transformed_pose_path(md5)
    if not pose_path.exists():
        raise FileNotFoundError(
            f"No pre-computed pose for {video_path} (md5={md5}) at {pose_path}"
        )
    return load_pose(pose_path)


def _probe_fps(video_path: Path) -> float:
    """Read the true frame rate with ffprobe; fall back to POSE_FPS."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
                str(video_path),
            ],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        num, _, den = out.partition("/")
        fps = float(num) / float(den) if den else float(num)
        return fps if fps > 0 else POSE_FPS
    except Exception:
        return POSE_FPS


def extract_holistic(video_path: Path, fps: Optional[float] = None) -> Pose:
    """MediaPipe holistic extraction (refined face) matching the corpus format."""
    import cv2  # local import: heavy, only needed for extraction
    from pose_format.utils.holistic import load_holistic

    if fps is None:
        fps = _probe_fps(video_path)

    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")

    h, w = frames[0].shape[:2]
    return load_holistic(
        frames,
        fps=fps,
        width=w,
        height=h,
        depth=w,
        progress=False,
        additional_holistic_config={"refine_face_landmarks": True},
    )
