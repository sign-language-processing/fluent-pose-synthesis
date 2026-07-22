"""Fluent (sign language) pose synthesis via improved concatenative stitching.

Public API:

    from fluent_pose_synthesis import concatenate_poses, StitchConfig
"""

from fluent_pose_synthesis.stitching.concatenate import (  # noqa: F401
    BASELINE,
    StitchConfig,
    concatenate_poses,
)

__all__ = ["concatenate_poses", "StitchConfig", "BASELINE"]
