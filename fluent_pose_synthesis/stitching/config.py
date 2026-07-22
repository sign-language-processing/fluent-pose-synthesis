"""Environment paths and constants for the DGS-based stitching pipeline.

All filesystem locations can be overridden with environment variables so the
code stays portable across machines.  Defaults point at the Nagish data mounts.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Data mounts ---------------------------------------------------------
# Raw DGS corpus / types (EAF, mp4, index.csv).
RAW_DATA_DIR = Path(os.environ.get("SIGN_RAW_DATA", "/mnt/nas/GCS/sign-raw-data-prod"))
DGS_CORPUS_DIR = RAW_DATA_DIR / "dgs-corpus"
DGS_TYPES_DIR = RAW_DATA_DIR / "dgs-types"

# Pre-computed holistic poses, keyed by md5 of the source video.
TRANSFORMED_VIDEOS_DIR = Path(
    os.environ.get("SIGN_TRANSFORMED_VIDEOS", "/mnt/r2/sign-transformed-data-prod/videos")
)

# --- Local cache ---------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = Path(os.environ.get("STITCH_CACHE", _REPO_ROOT / ".cache"))
TYPES_POSE_CACHE = CACHE_DIR / "types_poses"
EVAL_CACHE = CACHE_DIR / "eval"

for _d in (CACHE_DIR, TYPES_POSE_CACHE, EVAL_CACHE):
    _d.mkdir(parents=True, exist_ok=True)

# --- Segmentation model --------------------------------------------------
# Local safetensors dir for the sign-language-segmentation model (github.com/sign/segmentation).
SEGMENTATION_MODEL_DIR = Path(
    os.environ.get(
        "SEGMENTATION_MODEL_DIR",
        Path.home() / "shared/dev/sign/segmentation/sign_language_segmentation/dist/2026",
    )
)

# --- Pose format constants ----------------------------------------------
POSE_FPS = 50.0  # native DGS corpus / types frame rate
HOLISTIC_POINTS = 586
