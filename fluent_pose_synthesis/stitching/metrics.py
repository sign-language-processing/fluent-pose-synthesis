"""Metrics comparing a stitched reconstruction to the fluent reference.

Primary metric is DTWp from pose-evaluation (per-hand-keypoint DTW, order
invariant to length).  We also report length/tempo statistics, since the
dominant failure mode of concatenating citation forms is gross over-length.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from pose_format import Pose


@lru_cache(maxsize=1)
def _dtwp():
    from pose_evaluation.metrics.distance_metric import DistanceMetric
    from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure
    from pose_evaluation.metrics.pose_processors import (
        FillMaskedOrInvalidValuesPoseProcessor,
        GetHandsOnlyHolisticPoseProcessor,
        ReducePosesToCommonComponentsProcessor,
        TrimMeaninglessFramesPoseProcessor,
    )

    return DistanceMetric(
        name="DTWp",
        distance_measure=DTWDTAIImplementationDistanceMeasure(),
        pose_preprocessors=[
            TrimMeaninglessFramesPoseProcessor(),
            GetHandsOnlyHolisticPoseProcessor(),
            FillMaskedOrInvalidValuesPoseProcessor(masked_fill_value=10.0),
            ReducePosesToCommonComponentsProcessor(),
        ],
    )


def normalize(pose: Pose) -> Pose:
    """Shoulder-based normalization so poses from different videos are
    comparable in scale and position (required before DTW)."""
    from pose_format.utils.generic import pose_normalization_info

    return pose.normalize(pose_normalization_info(pose.header))


def dtwp(hypothesis: Pose, reference: Pose) -> float:
    """DTWp distance (lower = closer). Hands-only, length-normalized by DTW.

    Both poses are shoulder-normalized first; otherwise the distance is
    dominated by scale/offset differences between the two source videos.
    """
    return float(_dtwp().score(normalize(hypothesis), normalize(reference)))


def _hands_array(pose: Pose, fill: float = 10.0) -> np.ndarray:
    """Shoulder-normalized hand keypoints as a dense (frames, 42, 3) array."""
    p = normalize(pose)
    idx, offset = [], 0
    for comp in p.header.components:
        if comp.name in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
            idx.extend(range(offset, offset + len(comp.points)))
        offset += len(comp.points)
    data = np.ma.getdata(p.body.data)[:, 0, idx, :]
    mask = np.ma.getmaskarray(p.body.data)[:, 0, idx, :]
    return np.where(mask, fill, data).astype(np.float64)


def dtwp_norm(hypothesis: Pose, reference: Pose) -> float:
    """Path-step-normalized DTW on the joint hands trajectory.

    Raw DTW is a *cumulative* path distance, so it is biased toward shorter
    hypotheses (fewer terms).  Dividing the distance by the actual warping-path
    length yields the mean per-aligned-step distance — a length-fair measure of
    trajectory *shape* similarity.  Computed on the concatenated hand keypoints
    (one alignment per pose pair) so a single path length applies.
    """
    from dtaidistance import dtw, dtw_ndim

    hyp = _hands_array(hypothesis).reshape(-1, 42 * 3)
    ref = _hands_array(reference).reshape(-1, 42 * 3)
    if hyp.shape[0] < 2 or ref.shape[0] < 2:
        return float("nan")
    distance, paths = dtw_ndim.warping_paths_fast(hyp, ref)
    path = dtw.best_path(paths)
    return float(distance / max(1, len(path)))


def n_frames(pose: Pose) -> int:
    return int(pose.body.data.shape[0])


def hand_speed(pose: Pose) -> float:
    """Mean per-frame wrist displacement — a coarse tempo proxy (higher=faster)."""
    try:
        li = pose.header._get_point_index("POSE_LANDMARKS", "LEFT_WRIST")
        ri = pose.header._get_point_index("POSE_LANDMARKS", "RIGHT_WRIST")
    except Exception:
        return float("nan")
    data = np.ma.getdata(pose.body.data)
    wrists = data[:, 0, [li, ri], :2]
    if wrists.shape[0] < 2:
        return 0.0
    diffs = np.linalg.norm(np.diff(wrists, axis=0), axis=-1)
    return float(np.nanmean(diffs))


def score(hypothesis: Pose, reference: Pose) -> dict:
    """Full metric bundle for one (hypothesis, reference) pair."""
    hf, rf = n_frames(hypothesis), n_frames(reference)
    return {
        "dtwp": dtwp(hypothesis, reference),
        "dtwp_norm": dtwp_norm(hypothesis, reference),
        "hyp_frames": hf,
        "ref_frames": rf,
        "length_ratio": hf / rf if rf else float("nan"),
        "length_abs_err": abs(hf - rf),
    }


def aggregate(rows: list[dict]) -> dict:
    """Mean/median summary over per-sentence metric rows."""
    out = {}
    keys = ["dtwp", "dtwp_norm", "length_ratio", "length_abs_err", "hyp_frames", "ref_frames"]
    for k in keys:
        vals = np.array([r[k] for r in rows if r is not None and np.isfinite(r.get(k, np.nan))])
        if len(vals):
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_median"] = float(np.median(vals))
    out["n"] = len([r for r in rows if r is not None])
    return out
