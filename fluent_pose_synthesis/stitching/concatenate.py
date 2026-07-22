"""Configurable concatenative stitching of isolated poses into a sentence.

The default :class:`StitchConfig` reproduces
``spoken_to_signed.gloss_to_pose.concatenate_poses`` exactly.  Every technique
we experiment with (segmentation-based trimming, tempo compression, padding,
smoothing) is a toggle on the same config, so a change can be A/B-tested against
the baseline on identical inputs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.utils.generic import (
    correct_wrists,
    normalize_pose_size,
    pose_normalization_info,
    reduce_holistic,
)

from fluent_pose_synthesis.stitching.trim import trim_pose


@dataclass
class StitchConfig:
    # Preprocessing (baseline: all on)
    reduce_holistic: bool = True
    normalize: bool = True

    # Map every source sign to one canonical appearance before stitching, so the
    # sentence looks like a single signer (pose-anonymization). Off by default.
    anonymize: bool = False

    # Trimming of per-sign lead-in / lead-out
    trim: bool = True
    trim_method: str = "hand_raise"  # "hand_raise" | "segmentation" | "none"

    # Tempo: isolated citation forms are far slower than fluent signing.
    speed: float = 1.0  # >1 compresses every sign (frames -> frames/speed)
    frames_per_sign: Optional[int] = None  # if set, resample each sign to this length
    max_sign_frames: Optional[int] = None  # cap: only compress signs LONGER than this
    complexity_scaling: bool = False  # scale target length by sign motion complexity
    min_sign_frames: int = 4

    # Concatenation / smoothing
    connection_search: bool = True  # search for the closest transition frame
    connection_window: float = 0.30
    padding: float = 0.20  # seconds of interpolated transition between signs
    savgol: bool = True  # Savitzky-Golay temporal smoothing at the end

    # --- Techniques below (default off => baseline behaviour) ---------------
    # Transition between signs: "pad" (baseline) | "velocity" | "direct".
    transition: str = "pad"
    transition_min: int = 2          # min bridge frames (velocity/direct)
    transition_max: int = 16         # max bridge frames
    transition_speed: float = 1.0    # divides the velocity-matched bridge length

    # Butterworth low-pass temporal filter (Sign Stitching, BMVC 2024).
    butter: bool = False
    butter_cutoff: float = 6.0       # Hz
    butter_order: int = 4

    # Distribution alignment: learned constant shift of hand keypoints (x,y,z)
    # in shoulder-normalized space, to match the corpus hand-position mean.
    hand_shift: Optional[tuple] = None

    # Hand root/shape decomposition: bridge the wrist (root) directly while
    # holding handshape, minimizing spurious inter-sign hand movement.
    detach_hands: bool = False

    # Co-articulation: pull each sign spatially toward where the previous sign
    # ended, reducing inter-sign wrist travel (0 = off, 1 = fully connected).
    coarticulate: float = 0.0

    # Remove a hand that is idle for a sign (resting/low/undetected) so stitching
    # interpolates it from active neighbours instead of snapping to a rest pose.
    drop_inactive_hands: bool = False
    hand_active_ratio: float = 0.2  # drop a hand moving < this * the active hand's movement
    # How a removed hand is filled back across the gap: "linear" (constant-speed
    # slide, unnatural) or "ease_in" (hold near the previous pose, then a quick
    # preparation stroke into the next sign, like real signing).
    hand_transition: str = "linear"
    hand_ease_exp: float = 3.0  # ease-in exponent (higher = holds longer, faster end)


BASELINE = StitchConfig()


# --------------------------------------------------------------------------
# Tempo helpers
# --------------------------------------------------------------------------
def resample_pose(pose: Pose, target_frames: int) -> Pose:
    """Linearly resample a pose along time to ``target_frames`` frames."""
    data = pose.body.data  # masked (frames, people, points, dims)
    conf = pose.body.confidence
    n = data.shape[0]
    target_frames = max(1, int(target_frames))
    if n == target_frames or n < 2:
        return pose

    src = np.linspace(0.0, n - 1, num=n)
    dst = np.linspace(0.0, n - 1, num=target_frames)

    raw = np.ma.getdata(data).astype(np.float64)
    mask = np.ma.getmaskarray(data)
    filled = np.where(mask, np.nan, raw)

    out = np.empty((target_frames, *filled.shape[1:]), dtype=np.float64)
    flat = filled.reshape(n, -1)
    out_flat = out.reshape(target_frames, -1)
    for j in range(flat.shape[1]):
        out_flat[:, j] = np.interp(dst, src, flat[:, j])
    conf_out = np.empty((target_frames, *conf.shape[1:]), dtype=np.float64)
    cflat = conf.reshape(n, -1)
    cout = conf_out.reshape(target_frames, -1)
    for j in range(cflat.shape[1]):
        cout[:, j] = np.interp(dst, src, cflat[:, j])

    new_mask = np.isnan(out)
    out = np.ma.masked_array(np.nan_to_num(out), mask=new_mask)
    body = NumPyPoseBody(fps=pose.body.fps, data=out, confidence=conf_out)
    return Pose(header=pose.header, body=body)


def motion_complexity(pose: Pose) -> float:
    """Total wrist path length — a proxy for how much a sign actually moves."""
    try:
        li = pose.header._get_point_index("POSE_LANDMARKS", "LEFT_WRIST")
        ri = pose.header._get_point_index("POSE_LANDMARKS", "RIGHT_WRIST")
    except Exception:
        return 1.0
    data = np.ma.getdata(pose.body.data)[:, 0, [li, ri], :2]
    if data.shape[0] < 2:
        return 0.0
    return float(np.nansum(np.linalg.norm(np.diff(data, axis=0), axis=-1)))


def apply_tempo(pose: Pose, config: StitchConfig) -> Pose:
    n = pose.body.data.shape[0]
    if config.frames_per_sign is not None:
        target = config.frames_per_sign
        if config.complexity_scaling:
            # More movement -> proportionally more frames, centred on frames_per_sign.
            c = motion_complexity(pose)
            target = int(round(config.frames_per_sign * (0.5 + min(1.5, c / 3.0))))
    elif config.max_sign_frames is not None:
        # Cap only: leave already-short signs untouched, compress long ones.
        if n <= config.max_sign_frames:
            return pose
        target = config.max_sign_frames
    elif config.speed != 1.0:
        target = int(round(n / config.speed))
    else:
        return pose
    target = max(config.min_sign_frames, target)
    return resample_pose(pose, target)


# --------------------------------------------------------------------------
# Smoothing / concatenation (config-driven port of spoken-to-signed)
# --------------------------------------------------------------------------
def _pose_savgol_filter(pose: Pose) -> Pose:
    import scipy.signal

    [face] = [c for c in pose.header.components if c.name == "FACE_LANDMARKS"]
    face_range = range(
        pose.header._get_point_index("FACE_LANDMARKS", face.points[0]),
        pose.header._get_point_index("FACE_LANDMARKS", face.points[-1]),
    )
    _, _, points, dims = pose.body.data.shape
    for p in range(points):
        if p not in face_range:
            for d in range(dims):
                pose.body.data[:, 0, p, d] = scipy.signal.savgol_filter(pose.body.data[:, 0, p, d], 3, 1)
    return pose


def _butter_filter(pose: Pose, cutoff: float, order: int) -> Pose:
    """4th-order (default) low-pass Butterworth over time, per keypoint.

    Removes high-frequency jitter/seams while preserving sign motion
    (Sign Stitching, Walsh et al. BMVC 2024). Face is left untouched.
    """
    import scipy.signal

    fps = pose.body.fps
    nyq = fps / 2.0
    wn = min(max(cutoff / nyq, 1e-3), 0.99)
    b, a = scipy.signal.butter(order, wn, btype="low")

    face_range = set()
    for c in pose.header.components:
        if c.name == "FACE_LANDMARKS":
            base = pose.header._get_point_index("FACE_LANDMARKS", c.points[0])
            face_range = set(range(base, base + len(c.points)))
    n, _, points, dims = pose.body.data.shape
    padlen = 3 * max(len(a), len(b))
    if n <= padlen:
        return pose
    for p in range(points):
        if p in face_range:
            continue
        for d in range(dims):
            pose.body.data[:, 0, p, d] = scipy.signal.filtfilt(b, a, pose.body.data[:, 0, p, d])
    return pose


def _hand_indices(pose: Pose) -> list[int]:
    idx, offset = [], 0
    for c in pose.header.components:
        if c.name in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
            idx.extend(range(offset, offset + len(c.points)))
        offset += len(c.points)
    return idx


def _shoulder_width(pose: Pose) -> float:
    try:
        li = pose.header._get_point_index("POSE_LANDMARKS", "LEFT_SHOULDER")
        ri = pose.header._get_point_index("POSE_LANDMARKS", "RIGHT_SHOULDER")
    except Exception:
        return 1.0
    d = np.ma.getdata(pose.body.data)
    w = np.nanmean(np.linalg.norm(d[:, 0, li, :] - d[:, 0, ri, :], axis=-1))
    return float(w) if w and np.isfinite(w) else 1.0


def _apply_hand_shift(pose: Pose, shift) -> Pose:
    """Add (dx,dy,dz) — given in shoulder-normalized units — to every hand
    keypoint, converted into the pose's current scale (distribution match)."""
    idx = _hand_indices(pose)
    if not idx:
        return pose
    scale = _shoulder_width(pose)
    for d, s in enumerate(shift):
        if s:
            pose.body.data[:, 0, idx, d] = pose.body.data[:, 0, idx, d] + s * scale
    return pose


def _wrist_indices(pose: Pose):
    try:
        return (pose.header._get_point_index("POSE_LANDMARKS", "LEFT_WRIST"),
                pose.header._get_point_index("POSE_LANDMARKS", "RIGHT_WRIST"))
    except Exception:
        return None


def _boundary_velocity(pose: Pose, wi, ri, at_end: bool, k: int = 5) -> float:
    data = np.ma.getdata(pose.body.data)
    n = data.shape[0]
    if n < 2:
        return 0.0
    seg = data[max(0, n - k):] if at_end else data[:min(n, k)]
    w = seg[:, 0, [wi, ri], :]
    if w.shape[0] < 2:
        return 0.0
    return float(np.nanmean(np.linalg.norm(np.diff(w, axis=0), axis=-1)))


def _bridge_frames(end_frame: np.ndarray, start_frame: np.ndarray, n: int):
    """n interpolated frames strictly between two boundary frames (linear)."""
    if n <= 0:
        return None
    ts = np.linspace(0, 1, n + 2)[1:-1]  # exclude endpoints
    return np.stack([end_frame * (1 - t) + start_frame * t for t in ts])


def _coarticulate(poses: list[Pose], strength: float) -> None:
    """Translate each sign (all keypoints) toward where the previous ended,
    cutting the spurious 'return to neutral' travel between citation forms.

    Operates in place on already-trimmed poses.  strength=0 is a no-op; 1 makes
    consecutive dominant-hand wrists coincide across the join.
    """
    wr = _wrist_indices(poses[0])
    if wr is None or strength <= 0:
        return
    wi, ri = wr
    offset = np.zeros(poses[0].body.data.shape[-1])
    for i in range(1, len(poses)):
        prev = np.ma.getdata(poses[i - 1].body.data)
        cur = np.ma.getdata(poses[i].body.data)
        prev_end = prev[-1, 0, ri, :]
        cur_start = cur[0, 0, ri, :] + offset
        offset = offset + strength * (prev_end - cur_start)
        poses[i].body.data = poses[i].body.data + offset


def _fill_hand_gaps(body: NumPyPoseBody, header, exp: float) -> None:
    """Fill removed-hand gaps with an ease-in curve instead of a linear slide.

    For each hand keypoint, a masked span between two active frames is filled so
    the hand *holds* near its previous pose, then accelerates into the next sign
    (a preparation stroke). Leading/trailing gaps stay masked (hand absent until
    first needed). Operates in place; sets confidence on filled frames."""
    idx, offset = [], 0
    for c in header.components:
        if c.name in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"):
            idx.extend(range(offset, offset + len(c.points)))
        offset += len(c.points)
    if not idx:
        return
    data = np.ma.getdata(body.data)
    conf = body.confidence
    for k in idx:
        valid = np.where(conf[:, 0, k] > 0)[0]
        if len(valid) < 2:
            continue
        for a, b in zip(valid[:-1], valid[1:]):
            if b <= a + 1:
                continue
            for f in range(a + 1, b):
                t = (f - a) / (b - a)
                te = t ** exp
                data[f, 0, k, :] = data[a, 0, k, :] * (1 - te) + data[b, 0, k, :] * te
                conf[f, 0, k] = 1.0
    mask = np.ma.getmaskarray(body.data).copy()
    for k in idx:
        mask[:, 0, k, :] = (conf[:, 0, k] <= 0)[:, None]
    body.data = np.ma.array(data, mask=mask)


def _create_padding(seconds: float, example: Pose) -> NumPyPoseBody:
    fps = example.body.fps
    frames = int(seconds * fps)
    shape = example.body.data.shape
    return NumPyPoseBody(
        fps=fps,
        data=np.zeros((frames, shape[1], shape[2], shape[3])),
        confidence=np.zeros((frames, shape[1], shape[2])),
    )


def _find_best_connection_point(pose1: Pose, pose2: Pose, window: float):
    from scipy.spatial.distance import cdist

    p1 = math.ceil(min(window * pose1.body.fps, len(pose1.body.data) * window))
    p2 = math.ceil(min(window * pose2.body.fps, len(pose2.body.data) * window))
    last = pose1.body.data[len(pose1.body.data) - p1:]
    first = pose2.body.data[:p2]
    dist = cdist(last.reshape(len(last), -1), first.reshape(len(first), -1), "euclidean")
    idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
    return len(pose1.body.data) - p1 + idx[0], idx[1]


def _apply_connection_search(poses: list[Pose], config: StitchConfig) -> None:
    """Trim each pose in place to the frames that join most closely."""
    start = 0
    for i, pose in enumerate(poses):
        if i != len(poses) - 1 and config.connection_search:
            end, next_start = _find_best_connection_point(poses[i], poses[i + 1], config.connection_window)
        elif i != len(poses) - 1:
            end, next_start = len(pose.body.data), 0
        else:
            end, next_start = len(pose.body.data), None
        pose.body = pose.body[start:end]
        start = next_start


def _finish(body: NumPyPoseBody, header, config: StitchConfig) -> Pose:
    pose = Pose(header=header, body=body)
    if config.savgol:
        pose = _pose_savgol_filter(pose)
    if config.butter:
        pose = _butter_filter(pose, config.butter_cutoff, config.butter_order)
    return pose


def _smooth_concatenate(poses: list[Pose], config: StitchConfig, connection_done: bool = False) -> Pose:
    if len(poses) == 1:
        return _finish(poses[0].body, poses[0].header, config) if (config.savgol or config.butter) else poses[0]

    if not connection_done:
        _apply_connection_search(poses, config)

    if config.transition in ("velocity", "direct"):
        return _bridge_concatenate(poses, config)

    # Baseline "pad" transition: zero-padding + linear interpolation.
    padding = _create_padding(config.padding, poses[0])
    for pose in poses[:-1]:
        pose.body.data = np.concatenate((pose.body.data, padding.data))
        pose.body.confidence = np.concatenate((pose.body.confidence, padding.confidence))
    new_data = np.concatenate([p.body.data for p in poses])
    new_conf = np.concatenate([p.body.confidence for p in poses])
    body = NumPyPoseBody(fps=poses[0].body.fps, data=new_data, confidence=new_conf)
    if config.drop_inactive_hands and config.hand_transition == "ease_in":
        # Ease-in the removed-hand gaps before the linear fill handles the rest.
        _fill_hand_gaps(body, poses[0].header, config.hand_ease_exp)
    body = body.interpolate(kind="linear")
    return _finish(body, poses[0].header, config)


def _bridge_concatenate(poses: list[Pose], config: StitchConfig) -> Pose:
    """Insert an adaptive number of interpolated frames between signs.

    "direct":   a fixed small bridge (transition_min frames).
    "velocity": bridge length ∝ wrist gap / boundary velocity, so the transition
                moves at the signs' own speed instead of a fixed duration.
    """
    wr = _wrist_indices(poses[0])
    fps = poses[0].body.fps
    segs_data, segs_conf = [], []
    for i, pose in enumerate(poses):
        segs_data.append(np.ma.getdata(pose.body.data))
        segs_conf.append(pose.body.confidence)
        if i == len(poses) - 1:
            break
        end_f = np.ma.getdata(poses[i].body.data)[-1]
        start_f = np.ma.getdata(poses[i + 1].body.data)[0]
        if config.transition == "direct" or wr is None:
            n = config.transition_min
        else:
            wi, ri = wr
            gap = float(np.nanmean(np.linalg.norm(end_f[0, [wi, ri], :] - start_f[0, [wi, ri], :], axis=-1)))
            v = 0.5 * (_boundary_velocity(poses[i], wi, ri, at_end=True)
                       + _boundary_velocity(poses[i + 1], wi, ri, at_end=False))
            n = int(round(gap / v / config.transition_speed)) if v > 1e-6 else config.transition_min
            n = max(config.transition_min, min(config.transition_max, n))
        bridge = _bridge_frames(end_f, start_f, n)
        if bridge is not None:
            segs_data.append(bridge)
            segs_conf.append(np.ones((bridge.shape[0], *segs_conf[-1].shape[1:])))
    data = np.ma.array(np.concatenate(segs_data))
    conf = np.concatenate(segs_conf)
    body = NumPyPoseBody(fps=fps, data=data, confidence=conf)
    return _finish(body, poses[0].header, config)


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def _copy_pose(p: Pose) -> Pose:
    """Deep-ish copy so in-place steps never mutate lru-cached input poses."""
    data = np.ma.array(np.ma.getdata(p.body.data).copy(), mask=np.ma.getmaskarray(p.body.data).copy())
    body = NumPyPoseBody(fps=p.body.fps, data=data, confidence=p.body.confidence.copy())
    return Pose(header=p.header, body=body)


def _anonymize(pose: Pose) -> Pose:
    """Transfer a pose to the canonical mean appearance (pose-anonymization)."""
    from pose_anonymization.appearance import remove_appearance

    return remove_appearance(pose)


def _hand_activity(pose: Pose, side: str) -> tuple[float, float]:
    """(detected_fraction, mean wrist movement per detected frame) for a hand.

    A resting hand is *still and/or undetected* — not merely low — so activity is
    measured by motion, not height (a chest-level two-handed sign like HOUSE has
    both hands active but below the elbow)."""
    data = np.ma.getdata(pose.body.data)
    conf = pose.body.confidence
    wi = pose.header._get_point_index("POSE_LANDMARKS", f"{side}_WRIST")
    detected = conf[:, 0, wi] > 0
    det_frac = float(detected.mean())
    w = data[:, 0, wi, :2]
    step = np.linalg.norm(np.diff(w, axis=0), axis=-1)
    valid_step = detected[1:] & detected[:-1]
    move = float(np.mean(step[valid_step])) if valid_step.any() else 0.0
    return det_frac, move


def _drop_inactive_hands(pose: Pose, active_ratio: float) -> Pose:
    """Mask a hand that is idle for this sign (mostly undetected, or moving far
    less than the active hand — i.e. resting), so the stitcher interpolates it
    from neighbouring signs instead of freezing a resting hand into the sentence."""
    data = pose.body.data
    conf = pose.body.confidence
    if data.shape[0] == 0:
        return pose
    hand_comps = {"LEFT": "LEFT_HAND_LANDMARKS", "RIGHT": "RIGHT_HAND_LANDMARKS"}
    try:
        act = {s: _hand_activity(pose, s) for s in hand_comps}
    except Exception:
        return pose
    active_move = max(m for _, m in act.values()) or 1.0
    for side, comp in hand_comps.items():
        det_frac, move = act[side]
        inactive = det_frac < 0.3 or move < active_ratio * active_move
        if not inactive:
            continue
        offset = 0
        for c in pose.header.components:
            if c.name == comp:
                idx = list(range(offset, offset + len(c.points)))
                if isinstance(data, np.ma.MaskedArray):
                    data[:, 0, idx, :] = np.ma.masked
                conf[:, 0, idx] = 0.0
                break
            offset += len(c.points)
    return pose


def concatenate_poses(poses: list[Pose], config: StitchConfig = BASELINE) -> Pose:
    """Stitch isolated poses into one sequence under ``config``."""
    if not poses:
        raise ValueError("No poses to concatenate")
    poses = [_copy_pose(p) for p in poses]

    if config.anonymize:
        # Constant appearance across signs; must precede reduce_holistic
        # (anonymization needs the full holistic layout).
        poses = [_anonymize(p) for p in poses]
    if config.drop_inactive_hands:
        poses = [_drop_inactive_hands(p, config.hand_active_ratio) for p in poses]
    if config.reduce_holistic:
        poses = [reduce_holistic(p) for p in poses]
    if config.normalize:
        poses = [p.normalize(pose_normalization_info(p.header)) for p in poses]
    if config.trim:
        poses = [
            trim_pose(p, start=i > 0, end=i < len(poses) - 1, method=config.trim_method)
            for i, p in enumerate(poses)
        ]
    if config.frames_per_sign is not None or config.max_sign_frames is not None or config.speed != 1.0:
        poses = [apply_tempo(p, config) for p in poses]
    coart = config.coarticulate > 0
    if coart:
        _apply_connection_search(poses, config)
        _coarticulate(poses, config.coarticulate)

    pose = _smooth_concatenate(poses, config, connection_done=coart)
    pose = correct_wrists(pose)
    normalize_pose_size(pose)
    if config.hand_shift is not None:
        pose = _apply_hand_shift(pose, config.hand_shift)
    return pose
