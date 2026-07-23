"""Trim the non-signing lead-in / lead-out of an isolated pose.

Citation-form dictionary clips start and end with the hands at rest; those
frames inflate the stitched sentence and add unnatural pauses.  Two strategies:

* ``hand_raise`` — the spoken-to-signed heuristic: keep frames where the wrist
  is above the elbow (hands raised into signing space).
* ``segmentation`` — run the sign-segmentation model and keep the span it marks
  as an actual SIGN.  Falls back to ``hand_raise`` if the model is unavailable.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import NamedTuple, Optional

import numpy as np
from pose_format import Pose

from fluent_pose_synthesis.stitching.config import CACHE_DIR, SEGMENTATION_MODEL_DIR


class SigningBoundary(NamedTuple):
    start: Optional[int]
    end: Optional[int]


# --------------------------------------------------------------------------
# hand-raise heuristic (port of spoken_to_signed.gloss_to_pose.concatenate)
# --------------------------------------------------------------------------
def _hand_raise_boundary(pose: Pose, wrist_index: int, elbow_index: int) -> SigningBoundary:
    length = len(pose.body.data)
    wrist_exists = pose.body.confidence[:, 0, wrist_index] > 0
    first_non_zero = int(np.argmax(wrist_exists))
    last_non_zero = length - int(np.argmax(wrist_exists[::-1]))

    wrist_y = pose.body.data[:, 0, wrist_index, 1]
    elbow_y = pose.body.data[:, 0, elbow_index, 1]
    wrist_above = wrist_y < elbow_y
    if not np.any(wrist_above):
        return SigningBoundary(None, None)
    first_active = int(np.argmax(wrist_above))
    last_active = length - int(np.argmax(wrist_above[::-1]))
    return SigningBoundary(
        start=max(first_non_zero, first_active - 5),
        end=min(last_non_zero, last_active + 5),
    )


def _hand_raise_span(pose: Pose) -> Optional[tuple[int, int]]:
    firsts, lasts = [], []
    for hand in ("LEFT", "RIGHT"):
        wi = pose.header._get_point_index("POSE_LANDMARKS", f"{hand}_WRIST")
        ei = pose.header._get_point_index("POSE_LANDMARKS", f"{hand}_ELBOW")
        b = _hand_raise_boundary(pose, wi, ei)
        if b.start is not None:
            firsts.append(b.start)
        if b.end is not None:
            lasts.append(b.end)
    if not firsts:
        return None
    return min(firsts), max(lasts)


# --------------------------------------------------------------------------
# motion-threshold crop (Sign Stitching, Walsh et al. BMVC 2024)
# --------------------------------------------------------------------------
def _motion_threshold_span(pose: Pose, alpha: float = 0.2, min_frames: int = 4) -> Optional[tuple[int, int]]:
    """Drop lead-in/out until cumulative wrist motion exceeds ``alpha`` of total.

    Keeps the central, active portion of the clip: the first/last frame whose
    running displacement crosses ``alpha``× the sign's total path length.
    """
    try:
        li = pose.header._get_point_index("POSE_LANDMARKS", "LEFT_WRIST")
        ri = pose.header._get_point_index("POSE_LANDMARKS", "RIGHT_WRIST")
    except Exception:
        return None
    data = np.ma.getdata(pose.body.data)[:, 0, [li, ri], :2]
    if data.shape[0] < 2:
        return None
    step = np.nan_to_num(np.linalg.norm(np.diff(data, axis=0), axis=-1)).sum(axis=-1)  # per-frame motion
    total = step.sum()
    if total <= 0:
        return None
    cum = np.cumsum(step)
    start = int(np.argmax(cum >= alpha * total))
    end = int(np.argmax(cum >= (1 - alpha) * total)) + 1
    if end - start < min_frames:
        return None
    return start, end + 1


# --------------------------------------------------------------------------
# segmentation-model span
# --------------------------------------------------------------------------
_SEG_WARNED = False
_SEG_SPAN_CACHE_PATH = CACHE_DIR / "seg_spans.json"
_SEG_SPAN_CACHE: Optional[dict] = None


@lru_cache(maxsize=1)
def _seg_loader():
    from sign_language_segmentation.inference.adapters.model_store import ModelStore

    return ModelStore(model_dir=str(SEGMENTATION_MODEL_DIR), device="cpu")


def _pose_key(pose: Pose) -> str:
    return hashlib.md5(np.ascontiguousarray(np.ma.getdata(pose.body.data)).tobytes()).hexdigest()


def _load_seg_cache() -> dict:
    global _SEG_SPAN_CACHE
    if _SEG_SPAN_CACHE is None:
        try:
            _SEG_SPAN_CACHE = json.loads(_SEG_SPAN_CACHE_PATH.read_text())
        except Exception:
            _SEG_SPAN_CACHE = {}
    return _SEG_SPAN_CACHE


def _segmentation_span(pose: Pose) -> Optional[tuple[int, int]]:
    """First..last frame covered by the model's SIGN segments, or None.

    The sign-segmentation model reliably locates the active sign inside a
    citation clip, so it cuts the long resting lead-in/lead-out that the
    hand-raise heuristic leaves in.  Spans are disk-cached by pose hash so
    repeated experiments never re-run the model.
    """
    global _SEG_WARNED
    cache = _load_seg_cache()
    key = _pose_key(pose)
    if key in cache:
        span = cache[key]
        return tuple(span) if span is not None else None
    try:
        from sign_language_segmentation.inference.core.segmentation import segment_pose

        out = segment_pose(pose, model_loader=_seg_loader(), device="cpu")
        tiers = out[1] if isinstance(out, tuple) else out
        signs = tiers.get("SIGN", [])
        span = (min(s["start"] for s in signs), max(s["end"] for s in signs)) if signs else None
    except Exception as exc:  # noqa: BLE001
        if not _SEG_WARNED:
            print(f"[trim] segmentation unavailable ({exc}); falling back to hand_raise")
            _SEG_WARNED = True
        return None
    cache[key] = span
    try:
        _SEG_SPAN_CACHE_PATH.write_text(json.dumps(cache))
    except Exception:
        pass
    return span


# --------------------------------------------------------------------------
# public
# --------------------------------------------------------------------------
def trim_pose(pose: Pose, start: bool = True, end: bool = True, method: str = "hand_raise") -> Pose:
    """Return ``pose`` trimmed to its signing span.

    ``start`` / ``end`` gate whether each edge is trimmed (the first sign keeps
    its lead-in, the last keeps its lead-out, matching the baseline).
    """
    if method == "none" or len(pose.body.data) == 0:
        return pose

    span = None
    if method == "segmentation":
        span = _segmentation_span(pose)
    elif method == "motion":
        span = _motion_threshold_span(pose)
    if span is None:
        span = _hand_raise_span(pose)
    if span is None:
        return pose

    first, last = span
    if not start:
        first = 0
    if not end:
        last = len(pose.body.data)
    first = max(0, min(first, len(pose.body.data) - 1))
    last = max(first + 1, min(last, len(pose.body.data)))

    pose.body.data = pose.body.data[first:last]
    pose.body.confidence = pose.body.confidence[first:last]
    return pose
