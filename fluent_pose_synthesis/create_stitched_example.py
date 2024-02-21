from pathlib import Path

from pose_format import Pose
from spoken_to_signed.gloss_to_pose import concatenate_poses

glosses = [
    "DIFFERENT1^",
    "IMAGINATION1A^",
    "EQUAL1C^",
    "SMOOTH-OR-SLICK1^",
    "YOUNG1^",
    "HOUSE1A^"
]

poses_dir = Path(__file__).parent.parent / "assets" / "example" / "poses"

poses = []
for gloss in glosses:
    with open(poses_dir / f"{gloss}.pose", 'rb') as f:
        pose = Pose.read(f.read())
    poses.append(pose)

stitched_pose = concatenate_poses(poses)

with open(poses_dir / "stitched.pose", 'wb') as f:
    stitched_pose.write(f)

# visualize_pose -i stitched.pose -o stitched.mp4
# ffmpeg -i stitched.mp4 stitched.gif
