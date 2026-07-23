"""Smoke tests for the stitching core (no data mounts required)."""

import numpy as np
import numpy.ma as ma
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions

from fluent_pose_synthesis import BASELINE, StitchConfig
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


def test_butter_filter_preserves_shape():
    from fluent_pose_synthesis.stitching.concatenate import _butter_filter

    pose = _make_pose(60, points=5)
    out = _butter_filter(pose, cutoff=8.0, order=4)
    assert out.body.data.shape == pose.body.data.shape


def test_bridge_frame_count():
    from fluent_pose_synthesis.stitching.concatenate import _bridge_frames

    a = np.zeros((1, 5, 3))
    b = np.ones((1, 5, 3))
    assert _bridge_frames(a, b, 4).shape[0] == 4
    assert _bridge_frames(a, b, 0) is None


def test_fluent_preset_exists():
    from fluent_pose_synthesis.stitching.experiments import CONFIGS

    assert "fluent" in CONFIGS
    assert CONFIGS["fluent"].trim_method == "segmentation"
    assert CONFIGS["fluent"].butter


def test_copy_pose_is_independent():
    # Guards the cache-mutation bug: in-place steps must not touch the original.
    from fluent_pose_synthesis.stitching.concatenate import _copy_pose

    pose = _make_pose(20)
    original = np.ma.getdata(pose.body.data).copy()
    dup = _copy_pose(pose)
    dup.body.data[:] = dup.body.data + 5.0
    assert np.allclose(np.ma.getdata(pose.body.data), original)
