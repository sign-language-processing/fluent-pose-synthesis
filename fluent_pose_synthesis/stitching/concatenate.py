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


def _smooth_concatenate(poses: list[Pose], config: StitchConfig) -> Pose:
    if len(poses) == 1:
        return poses[0]

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

    padding = _create_padding(config.padding, poses[0])
    for pose in poses[:-1]:
        pose.body.data = np.concatenate((pose.body.data, padding.data))
        pose.body.confidence = np.concatenate((pose.body.confidence, padding.confidence))

    new_data = np.concatenate([p.body.data for p in poses])
    new_conf = np.concatenate([p.body.confidence for p in poses])
    body = NumPyPoseBody(fps=poses[0].body.fps, data=new_data, confidence=new_conf)
    body = body.interpolate(kind="linear")
    pose = Pose(header=poses[0].header, body=body)
    return _pose_savgol_filter(pose) if config.savgol else pose


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def concatenate_poses(poses: list[Pose], config: StitchConfig = BASELINE) -> Pose:
    """Stitch isolated poses into one sequence under ``config``."""
    if not poses:
        raise ValueError("No poses to concatenate")
    poses = list(poses)

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

    pose = _smooth_concatenate(poses, config)
    pose = correct_wrists(pose)
    normalize_pose_size(pose)
    return pose
