"""Diagnose where the stitched-vs-reference DTWp error comes from.

Aggregates per-pair diagnostics over an eval sample for a given stitch config,
so we can target the biggest error source instead of guessing.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json

import numpy as np

from fluent_pose_synthesis.stitching import dgs_corpus as dc
from fluent_pose_synthesis.stitching import harness as H
from fluent_pose_synthesis.stitching import metrics as M
from fluent_pose_synthesis.stitching.experiments import CONFIGS, stitch_fn


def run(config_name: str, n: int, seed: int) -> dict:
    sents = H.build_eval_set(n=n, seed=seed)
    fn = stitch_fn(CONFIGS[config_name])
    rows = []
    for s in sents:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                hyp = H.reconstruct(s, fn)
                ref = dc.sentence_pose(s)
                d = M.diagnose(hyp, ref)
        except Exception:
            continue
        rows.append(d)

    keys = sorted({k for r in rows for k in r})
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])
        if len(vals):
            agg[k] = {"mean": round(float(vals.mean()), 4), "median": round(float(np.median(vals)), 4)}
    agg["_config"] = config_name
    agg["_n"] = len(rows)
    return agg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="baseline")
    p.add_argument("--n", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    print(json.dumps(run(args.config, args.n, args.seed), indent=2))


if __name__ == "__main__":
    main()
