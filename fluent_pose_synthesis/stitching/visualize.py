"""Render poses to GIFs with a consistent, anonymized, well-framed skeleton.

Each pose is mapped to the canonical mean signer (pose-anonymization), then
shoulder-normalized and rescaled with ``normalize_pose_size`` (the same rescale
the stitching pipeline uses), so several clips render at one identical scale and
appearance — ideal for side-by-side comparisons.

    python -m fluent_pose_synthesis.stitching.visualize a.pose b.pose --out-dir gifs/
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer
from pose_format.utils.generic import normalize_pose_size, pose_normalization_info, reduce_holistic


def frame_pose(pose: Pose, size: int = 512, margin: float = 0.72, anonymize: bool = True) -> Pose:
    """Anonymize (optional), shoulder-normalize, rescale to a ``size`` canvas,
    then shrink toward centre by ``margin`` so raised/extended arms don't clip."""
    if anonymize:
        from pose_anonymization.appearance import remove_appearance

        pose = remove_appearance(pose)
    pose = reduce_holistic(pose)
    pose = pose.normalize(pose_normalization_info(pose.header))
    normalize_pose_size(pose, target_width=size)
    c = size / 2
    pose.body.data[..., :2] = (pose.body.data[..., :2] - c) * margin + c
    return pose


def render_gif(pose: Pose, out_gif: Path, size: int = 512, fps: int = 25,
               width: int = 360, anonymize: bool = True) -> None:
    pose = frame_pose(pose, size=size, anonymize=anonymize)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
        viz = PoseVisualizer(pose)
        viz.save_video(tmp.name, viz.draw())
        pal = str(out_gif) + ".pal.png"
        vf = f"fps={fps},scale={width}:-1:flags=lanczos"
        subprocess.run(["ffmpeg", "-y", "-i", tmp.name, "-vf", f"{vf},palettegen", pal],
                       check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", tmp.name, "-i", pal, "-lavfi",
                        f"{vf}[x];[x][1:v]paletteuse", str(out_gif)],
                       check=True, capture_output=True)
        Path(pal).unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("poses", nargs="+", help=".pose files to render")
    p.add_argument("--out-dir", default=".")
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--no-anonymize", action="store_true")
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in args.poses:
        pose = Pose.read(Path(path).read_bytes())
        gif = out / (Path(path).stem + ".gif")
        render_gif(pose, gif, size=args.size, anonymize=not args.no_anonymize)
        print(f"{path} -> {gif} ({pose.body.data.shape[0]} frames)")


if __name__ == "__main__":
    main()
