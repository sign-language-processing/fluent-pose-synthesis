"""Trim the non-signing lead-in / lead-out of an isolated pose.

Citation-form dictionary clips start and end with the hands at rest; those
frames inflate the stitched sentence and add unnatural pauses.  Two strategies:

* ``hand_raise`` — the spoken-to-signed heuristic: keep frames where the wrist
  is above the elbow (hands raised into signing space).
* ``segmentation`` — run the sign-segmentation model and keep the span it marks
  as an actual SIGN.  Falls back to ``hand_raise`` if the model is unavailable.
"""

from __future__ import annotations

from functools import lru_cache
from typing import NamedTuple, Optional

import numpy as np
from pose_format import Pose

from fluent_pose_synthesis.stitching.config import SEGMENTATION_MODEL_DIR


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
# segmentation-model span
# --------------------------------------------------------------------------
_SEG_WARNED = False


@lru_cache(maxsize=1)
def _seg_loader():
    from sign_language_segmentation.inference.adapters.model_store import ModelStore

    return ModelStore(model_dir=str(SEGMENTATION_MODEL_DIR), device="cpu")


def _segmentation_span(pose: Pose) -> Optional[tuple[int, int]]:
    """First..last frame covered by the model's SIGN segments, or None.

    The sign-segmentation model reliably locates the active sign inside a
    citation clip, so it cuts the long resting lead-in/lead-out that the
    hand-raise heuristic leaves in.
    """
    global _SEG_WARNED
    try:
        from sign_language_segmentation.inference.core.segmentation import segment_pose

        out = segment_pose(pose, model_loader=_seg_loader(), device="cpu")
        tiers = out[1] if isinstance(out, tuple) else out
        signs = tiers.get("SIGN", [])
        if not signs:
            return None
        return min(s["start"] for s in signs), max(s["end"] for s in signs)
    except Exception as exc:  # noqa: BLE001
        if not _SEG_WARNED:
            print(f"[trim] segmentation unavailable ({exc}); falling back to hand_raise")
            _SEG_WARNED = True
        return None


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
