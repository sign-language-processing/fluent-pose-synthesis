"""Command-line entry point for the fluent stitching pipeline.

    fluent-stitch stitch   --glosses HAUS1A WISSEN2B ... --out out.pose
    fluent-stitch evaluate --n 40 --seed 0 --config seg_pad00
    fluent-stitch analyze  --n 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fluent_pose_synthesis.stitching import dgs_types as dt
from fluent_pose_synthesis.stitching.concatenate import concatenate_poses
from fluent_pose_synthesis.stitching.experiments import CONFIGS


def _cmd_stitch(args: argparse.Namespace) -> None:
    from pose_format import Pose

    poses = []
    for item in args.glosses:
        if item.endswith(".pose"):
            with open(item, "rb") as f:
                poses.append(Pose.read(f.read()))
        else:
            pose = dt.get_pose(item)
            if pose is None:
                print(f"  skip {item}: not in DGS Types dictionary")
                continue
            poses.append(pose)
    if not poses:
        raise SystemExit("no poses to stitch")
    config = CONFIGS[args.config]
    result = concatenate_poses(poses, config)
    with open(args.out, "wb") as f:
        result.write(f)
    print(f"wrote {args.out} ({result.body.data.shape[0]} frames, config={args.config})")


def _cmd_evaluate(args: argparse.Namespace) -> None:
    from fluent_pose_synthesis.stitching.experiments import run

    run([args.config] if args.config != "all" else list(CONFIGS), args.n, args.seed, Path(args.out))


def _cmd_analyze(args: argparse.Namespace) -> None:
    import json

    from fluent_pose_synthesis.stitching.analysis import analyze

    print(json.dumps(analyze(args.n, args.seed), indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(prog="fluent-stitch", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("stitch", help="stitch DGS-Types forms (or .pose files) into a sentence")
    s.add_argument("--glosses", nargs="+", required=True, help="gloss names or .pose paths, in order")
    s.add_argument("--out", default="stitched.pose")
    s.add_argument("--config", default="fluent", choices=list(CONFIGS))
    s.set_defaults(func=_cmd_stitch)

    e = sub.add_parser("evaluate", help="evaluate a config against the DGS corpus")
    e.add_argument("--n", type=int, default=40)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--config", default="all")
    e.add_argument("--out", default="autoresearch/cli_eval")
    e.set_defaults(func=_cmd_evaluate)

    a = sub.add_parser("analyze", help="characterize corpus vs. reconstruction")
    a.add_argument("--n", type=int, default=40)
    a.add_argument("--seed", type=int, default=0)
    a.set_defaults(func=_cmd_analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
