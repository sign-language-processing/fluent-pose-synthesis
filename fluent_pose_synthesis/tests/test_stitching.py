"""Smoke tests for the stitching core (no data mounts required)."""

import numpy as np
import numpy.ma as ma
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

from fluent_pose_synthesis import BASELINE, StitchConfig, concatenate_poses
from fluent_pose_synthesis.stitching.concatenate import motion_complexity, resample_pose


def _make_pose(frames: int, points: int = 5, fps: float = 50.0) -> Pose:
    comp = PoseHeaderComponent(
        name="BODY",
        points=[f"p{i}" for i in range(points)],
        limbs=[],
        colors=[(0, 0, 0)],
        point_format="XYZ",
    )
    header = PoseHeader(version=0.1, dimensions=PoseHeaderDimensions(1, 1, 1), components=[comp])
    data = ma.array(np.random.RandomState(0).rand(frames, 1, points, 3).astype(np.float32))
    conf = np.ones((frames, 1, points), dtype=np.float32)
    return Pose(header, NumPyPoseBody(fps=fps, data=data, confidence=conf))


def test_resample_changes_length():
    pose = _make_pose(100)
    out = resample_pose(pose, 20)
    assert out.body.data.shape[0] == 20
    assert out.body.data.shape[1:] == pose.body.data.shape[1:]


def test_resample_noop_when_equal():
    pose = _make_pose(30)
    assert resample_pose(pose, 30).body.data.shape[0] == 30


def test_motion_complexity_nonnegative():
    # No POSE_LANDMARKS component -> falls back to 1.0, still finite.
    assert np.isfinite(motion_complexity(_make_pose(10)))


def test_baseline_config_matches_spoken_to_signed_defaults():
    assert BASELINE.trim_method == "hand_raise"
    assert BASELINE.padding == 0.20
    assert BASELINE.speed == 1.0
    assert BASELINE.reduce_holistic and BASELINE.normalize and BASELINE.trim and BASELINE.savgol


def test_stitch_config_toggles():
    cfg = StitchConfig(trim_method="segmentation", padding=0.0, max_sign_frames=25)
    assert cfg.trim_method == "segmentation"
    assert cfg.padding == 0.0
    assert cfg.max_sign_frames == 25
