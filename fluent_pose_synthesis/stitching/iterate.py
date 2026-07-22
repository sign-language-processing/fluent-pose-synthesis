"""Hypothesis-driven iteration driver.

A "batch" is a list of (label, hypothesis, StitchConfig). Each config is
evaluated on a fixed corpus sample and appended to ``results.tsv`` in the run
directory. Primary objective: ``dtwp_matched`` (shape at matched length);
secondary: length_ratio→1.0 and the velocity/acceleration distribution
distances (vel_w, acc_w).
"""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

from fluent_pose_synthesis.stitching import harness as H
from fluent_pose_synthesis.stitching import metrics as M
from fluent_pose_synthesis.stitching.concatenate import concatenate_poses

_SENTS_CACHE: dict = {}

METRIC_COLS = ["dtwp_clean", "dtwp_matched", "dtwp", "length_ratio", "vel_w", "acc_w", "pos_std_err", "vel_mean_err", "still_frac"]


def get_sents(n: int, seed: int):
    key = (n, seed)
    if key not in _SENTS_CACHE:
        _SENTS_CACHE[key] = H.build_eval_set(n=n, seed=seed)
    return _SENTS_CACHE[key]


def _stitch_fn(config):
    def fn(poses):
        with contextlib.redirect_stdout(io.StringIO()):
            return concatenate_poses(poses, config)
    return fn


def evaluate(config, n: int, seed: int) -> dict:
    rows = H.evaluate(get_sents(n, seed), _stitch_fn(config))
    return M.aggregate([r for r in rows if r and "error" not in r])


def run_batch(batch: list[dict], run_dir: str, n: int = 30, seed: int = 0) -> list[dict]:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    tsv = run / "results.tsv"
    header = ["label", "hypothesis", *[f"{c}_mean" for c in METRIC_COLS]]
    if not tsv.exists():
        tsv.write_text("\t".join(header) + "\n")

    results = []
    for item in batch:
        agg = evaluate(item["config"], n, seed)
        row = {"label": item["label"], "hypothesis": item["hypothesis"], **agg}
        results.append(row)
        with open(tsv, "a") as f:
            f.write("\t".join([item["label"], item["hypothesis"],
                               *[f"{agg.get(f'{c}_mean', float('nan')):.4f}" for c in METRIC_COLS]]) + "\n")
        print(f"  {item['label']:22s} clean={agg.get('dtwp_clean_mean', float('nan')):6.3f} "
              f"dtwpM={agg.get('dtwp_matched_mean', float('nan')):6.2f} "
              f"len={agg.get('length_ratio_mean', float('nan')):4.2f} "
              f"velW={agg.get('vel_w_mean', float('nan')):.4f} "
              f"still={agg.get('still_frac_mean', float('nan')):.2f}")
    return results
