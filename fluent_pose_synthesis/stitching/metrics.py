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


def dtwp_matched(hypothesis: Pose, reference: Pose) -> float:
    """DTWp after resampling the hypothesis to the reference length.

    Removes the length confound entirely, isolating trajectory *shape* quality —
    the hard target. (Peeks at reference length; a measurement aid, not a
    deployable step.)"""
    from fluent_pose_synthesis.stitching.concatenate import resample_pose

    hyp = resample_pose(hypothesis, n_frames(reference))
    return dtwp(hyp, reference)


def _interp_masked(arr: np.ma.MaskedArray) -> np.ndarray:
    """Linearly interpolate masked values along time, per keypoint/dim."""
    data = np.ma.getdata(arr).astype(np.float64).copy()
    mask = np.ma.getmaskarray(arr)
    n = data.shape[0]
    t = np.arange(n)
    flat_d = data.reshape(n, -1)
    flat_m = mask.reshape(n, -1)
    for j in range(flat_d.shape[1]):
        valid = ~flat_m[:, j]
        if valid.sum() >= 2:
            flat_d[:, j] = np.interp(t, t[valid], flat_d[valid, j])
        elif valid.sum() == 1:
            flat_d[:, j] = flat_d[valid, j][0]
        else:
            flat_d[:, j] = 0.0
    return flat_d.reshape(data.shape)


def dtwp_clean(hypothesis: Pose, reference: Pose) -> float:
    """Shape-at-matched-length DTW with masked keypoints interpolated (not
    filled with 10.0), so it reflects genuine trajectory quality rather than
    detection mismatches. Joint-hands, path-length normalized."""
    from dtaidistance import dtw, dtw_ndim

    hyp = _interp_masked(_hands_masked(hypothesis)).reshape(n_frames(hypothesis), -1)
    ref = _interp_masked(_hands_masked(reference)).reshape(n_frames(reference), -1)
    if hyp.shape[0] < 2 or ref.shape[0] < 2:
        return float("nan")
    # Resample hyp to ref length to isolate shape from length.
    src = np.linspace(0, hyp.shape[0] - 1, hyp.shape[0])
    dst = np.linspace(0, hyp.shape[0] - 1, ref.shape[0])
    hyp = np.stack([np.interp(dst, src, hyp[:, j]) for j in range(hyp.shape[1])], axis=1)
    distance, paths = dtw_ndim.warping_paths_fast(np.ascontiguousarray(hyp), np.ascontiguousarray(ref))
    path = dtw.best_path(paths)
    return float(distance / max(1, len(path)))


def masked_fraction_hands(pose: Pose) -> float:
    """Fraction of hand-keypoint values that are masked/undetected."""
    p = normalize(pose)
    idx, offset = [], 0
    for comp in p.header.components:
        if comp.name in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
            idx.extend(range(offset, offset + len(comp.points)))
        offset += len(comp.points)
    mask = np.ma.getmaskarray(p.body.data)[:, 0, idx, :]
    return float(mask.mean()) if mask.size else 0.0


def diagnose(hypothesis: Pose, reference: Pose) -> dict:
    """Per-pair breakdown of where the hyp/ref hand difference comes from."""
    from dtaidistance import dtw_ndim

    hyp = _hands_array(hypothesis)  # (frames,42,3), masked filled with 10.0
    ref = _hands_array(reference)
    out = {
        "hyp_masked_frac": masked_fraction_hands(hypothesis),
        "ref_masked_frac": masked_fraction_hands(reference),
    }
    if hyp.shape[0] < 2 or ref.shape[0] < 2:
        return out
    # Per-keypoint DTW distance (which of the 42 keypoints dominate).
    per_kp = np.array([dtw_ndim.distance_fast(hyp[:, k, :], ref[:, k, :]) for k in range(42)])
    out["dtw_left_hand"] = float(per_kp[:21].mean())
    out["dtw_right_hand"] = float(per_kp[21:].mean())
    out["dtw_wrist_lr"] = float((per_kp[0] + per_kp[21]) / 2)   # keypoint 0 = wrist
    out["dtw_fingers"] = float(np.concatenate([per_kp[1:21], per_kp[22:]]).mean())
    # Systematic position offset (mean hand centroid), per axis.
    hyp_c = hyp.reshape(-1, 3).mean(0)
    ref_c = ref.reshape(-1, 3).mean(0)
    out["pos_offset_x"] = float(hyp_c[0] - ref_c[0])
    out["pos_offset_y"] = float(hyp_c[1] - ref_c[1])
    out["pos_offset_z"] = float(hyp_c[2] - ref_c[2])
    return out


def _hands_masked(pose: Pose):
    """Shoulder-normalized hand keypoints as a masked (frames, 42, 3) array
    (masked = undetected), so statistics can ignore missing keypoints instead
    of being corrupted by the DTWp 10.0 fill value."""
    p = normalize(pose)
    idx, offset = [], 0
    for comp in p.header.components:
        if comp.name in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
            idx.extend(range(offset, offset + len(comp.points)))
        offset += len(comp.points)
    data = np.ma.getdata(p.body.data)[:, 0, idx, :]
    mask = np.ma.getmaskarray(p.body.data)[:, 0, idx, :]
    return np.ma.array(data, mask=mask)


def hand_velocity_series(pose: Pose) -> np.ndarray:
    """Per-frame mean hand-keypoint displacement (shoulder-normalized), masked
    keypoints ignored. Returns finite per-frame speeds only."""
    h = _hands_masked(pose)
    if h.shape[0] < 2:
        return np.zeros(0)
    disp = np.sqrt((np.diff(h, axis=0) ** 2).sum(axis=-1))  # (frames-1, 42), masked
    speed = disp.mean(axis=1)  # masked mean over keypoints
    arr = np.ma.filled(speed, np.nan)
    return arr[np.isfinite(arr)]


def hand_accel_series(pose: Pose) -> np.ndarray:
    v = hand_velocity_series(pose)
    return np.abs(np.diff(v)) if len(v) >= 2 else np.zeros(0)


def hand_jerk_series(pose: Pose) -> np.ndarray:
    a = hand_accel_series(pose)
    return np.abs(np.diff(a)) if len(a) >= 2 else np.zeros(0)


def hand_position_axis(pose: Pose, axis: int) -> np.ndarray:
    """Flattened, mask-filtered hand-keypoint positions on one axis."""
    h = _hands_masked(pose)[:, :, axis]
    arr = np.ma.filled(h.reshape(-1), np.nan)
    return arr[np.isfinite(arr)]


def distribution_distance(hypothesis: Pose, reference: Pose) -> dict:
    """How far the hypothesis's motion/position *distributions* are from the
    reference's — complements DTWp (which only scores aligned trajectories).

    * vel_w / acc_w : 1-Wasserstein distance between velocity / acceleration
      distributions (tempo + jitter, alignment-free).
    * pos_std_err   : |std(hyp) - std(ref)| of hand position (spatial spread).
    """
    from scipy.stats import wasserstein_distance

    def w(a, b):
        return float(wasserstein_distance(a, b)) if len(a) and len(b) else float("nan")

    out = {}
    hv, rv = hand_velocity_series(hypothesis), hand_velocity_series(reference)
    ha, ra = hand_accel_series(hypothesis), hand_accel_series(reference)
    hj, rj = hand_jerk_series(hypothesis), hand_jerk_series(reference)
    out["vel_w"] = w(hv, rv)
    out["acc_w"] = w(ha, ra)
    out["jerk_w"] = w(hj, rj)
    out["posx_w"] = w(hand_position_axis(hypothesis, 0), hand_position_axis(reference, 0))
    out["posy_w"] = w(hand_position_axis(hypothesis, 1), hand_position_axis(reference, 1))
    hp = _hands_masked(hypothesis).reshape(-1, 3)
    rp = _hands_masked(reference).reshape(-1, 3)
    hp_std = np.ma.std(hp, axis=0).mean()
    rp_std = np.ma.std(rp, axis=0).mean()
    out["pos_std_err"] = float(abs(hp_std - rp_std))
    out["vel_mean_err"] = float(abs(np.median(hv) - np.median(rv))) if len(hv) and len(rv) else float("nan")
    out["still_frac"] = float(np.mean(hv < 0.01)) if len(hv) else float("nan")
    out["still_frac_err"] = (
        float(abs(np.mean(hv < 0.01) - np.mean(rv < 0.01))) if len(hv) and len(rv) else float("nan")
    )
    return out


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
    row = {
        "dtwp": dtwp(hypothesis, reference),
        "dtwp_norm": dtwp_norm(hypothesis, reference),
        "dtwp_matched": dtwp_matched(hypothesis, reference),
        "dtwp_clean": dtwp_clean(hypothesis, reference),
        "hyp_frames": hf,
        "ref_frames": rf,
        "length_ratio": hf / rf if rf else float("nan"),
        "length_abs_err": abs(hf - rf),
    }
    row.update(distribution_distance(hypothesis, reference))
    return row


def aggregate(rows: list[dict]) -> dict:
    """Mean/median summary over per-sentence metric rows."""
    out = {}
    keys = ["dtwp", "dtwp_norm", "dtwp_matched", "dtwp_clean", "length_ratio", "length_abs_err",
            "hyp_frames", "ref_frames", "vel_w", "acc_w", "jerk_w", "posx_w", "posy_w",
            "pos_std_err", "vel_mean_err", "still_frac", "still_frac_err", "emb_cos"]
    for k in keys:
        vals = np.array([r[k] for r in rows if r is not None and np.isfinite(r.get(k, np.nan))])
        if len(vals):
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_median"] = float(np.median(vals))
    out["n"] = len([r for r in rows if r is not None])
    return out
