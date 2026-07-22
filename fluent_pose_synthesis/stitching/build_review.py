"""Render a review dataset: for N DGS-corpus sentences, produce small MP4s of
each constituent sign plus the naive / fluent / gold pose, base64-embedded into
one JSON for a self-contained review page."""

from __future__ import annotations

import base64
import contextlib
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from pose_format import Pose
from pose_format.pose_visualizer import PoseVisualizer

from fluent_pose_synthesis.stitching import dgs_corpus as dc
from fluent_pose_synthesis.stitching import dgs_types as dt
from fluent_pose_synthesis.stitching import harness as H
from fluent_pose_synthesis.stitching.concatenate import StitchConfig, concatenate_poses
from fluent_pose_synthesis.stitching.experiments import CONFIGS
from fluent_pose_synthesis.stitching.visualize import frame_pose

NAIVE = StitchConfig(anonymize=True)  # spoken-to-signed baseline + consistent signer
FLUENT = CONFIGS["fluent"]


def _mp4_data_uri(pose: Pose, size: int, fps: int, anonymize: bool) -> str:
    pose = frame_pose(pose, size=size, anonymize=anonymize)
    with tempfile.TemporaryDirectory() as d:
        raw = f"{d}/raw.mp4"
        out = f"{d}/out.mp4"
        viz = PoseVisualizer(pose)
        viz.save_video(raw, viz.draw())
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw, "-r", str(fps), "-vf", f"scale={size}:-2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out],
            check=True, capture_output=True,
        )
        data = Path(out).read_bytes()
    return "data:video/mp4;base64," + base64.b64encode(data).decode("ascii")


def build(n: int, seed: int) -> list[dict]:
    sents = H.build_eval_set(n=n, seed=seed, min_coverage=0.6, min_lexical=4, max_lexical=9)
    out = []
    for i, s in enumerate(sents):
        print(f"[{i+1}/{len(sents)}] {s.key}: {s.english[:60]}", flush=True)
        with contextlib.redirect_stdout(io.StringIO()):
            sources = H.gloss_sources(s)
            gold = dc.sentence_pose(s)
            naive = concatenate_poses([g.pose for g in sources], NAIVE)
            fluent = concatenate_poses([g.pose for g in sources], FLUENT)

        signs = []
        for g in s.glosses:
            if not g.is_lexical:
                continue
            pose = dt.get_pose(g.gloss)
            if pose is None or pose.body.data.shape[0] == 0:
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                uri = _mp4_data_uri(pose, size=150, fps=25, anonymize=False)
            signs.append({"gloss": g.gloss, "meaning": g.english, "video": uri})

        with contextlib.redirect_stdout(io.StringIO()):
            naive_uri = _mp4_data_uri(naive, size=240, fps=30, anonymize=True)
            fluent_uri = _mp4_data_uri(fluent, size=240, fps=30, anonymize=True)
            gold_uri = _mp4_data_uri(gold, size=240, fps=30, anonymize=True)

        out.append({
            "english": s.english, "german": s.german,
            "signs": signs,
            "naive": naive_uri, "fluent": fluent_uri, "gold": gold_uri,
            "frames": {"naive": naive.body.data.shape[0], "fluent": fluent.body.data.shape[0],
                       "gold": gold.body.data.shape[0]},
        })
    return out


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    out_path = sys.argv[3] if len(sys.argv) > 3 else "review_data.json"
    data = build(n, seed)
    Path(out_path).write_text(json.dumps(data))
    total = len(json.dumps(data)) / 1e6
    print(f"wrote {out_path}: {len(data)} sentences, {total:.1f} MB")


if __name__ == "__main__":
    main()
