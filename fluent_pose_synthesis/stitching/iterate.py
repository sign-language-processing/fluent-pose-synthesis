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

import numpy as np

from fluent_pose_synthesis.stitching import dgs_corpus as dc
from fluent_pose_synthesis.stitching import harness as H
from fluent_pose_synthesis.stitching import metrics as M
from fluent_pose_synthesis.stitching.concatenate import concatenate_poses

_SENTS_CACHE: dict = {}

METRIC_COLS = ["vel_w", "acc_w", "jerk_w", "posy_w", "still_frac_err", "emb_cos", "emb_fpd",
               "dtwp_matched", "length_ratio"]


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


def evaluate(config, n: int, seed: int, embed: bool = False) -> dict:
    """Evaluate a config on the fixed sample. When ``embed`` and the SignCLIP
    service is up, also compute per-sentence embedding cosine distance and the
    Fréchet Pose Distance (distribution-level) over the batch."""
    from fluent_pose_synthesis.stitching import signclip as SC

    fn = _stitch_fn(config)
    sents = get_sents(n, seed)
    do_embed = embed and SC.health()
    rows, hyp_embs, ref_embs = [], [], []
    for s in sents:
        try:
            hyp = H.reconstruct(s, fn)
            if hyp is None:
                continue
            ref = dc.sentence_pose(s)
            row = M.score(hyp, ref)
        except Exception as exc:  # noqa: BLE001
            rows.append({"error": str(exc)})
            continue
        if do_embed:
            try:  # embed failure must not drop the kinematic metrics
                he, re = SC.embed_pose(hyp), SC.embed_pose(ref)
                row["emb_cos"] = SC.cosine_distance(he, re)
                hyp_embs.append(he)
                ref_embs.append(re)
            except Exception:  # noqa: BLE001
                pass
        rows.append(row)
    agg = M.aggregate([r for r in rows if r and "error" not in r])
    if do_embed and len(hyp_embs) >= 2:
        agg["emb_fpd"] = SC.frechet_distance(np.array(hyp_embs), np.array(ref_embs))
    return agg


def _get(agg, col):
    # emb_fpd is a scalar (not averaged); others are *_mean
    return agg.get(col, float("nan")) if col == "emb_fpd" else agg.get(f"{col}_mean", float("nan"))


def run_batch(batch: list[dict], run_dir: str, n: int = 30, seed: int = 0, embed: bool = False) -> list[dict]:
    run = Path(run_dir)
    run.mkdir(parents=True, exist_ok=True)
    tsv = run / "results.tsv"
    header = ["label", "hypothesis", *METRIC_COLS]
    if not tsv.exists():
        tsv.write_text("\t".join(header) + "\n")

    results = []
    for item in batch:
        agg = evaluate(item["config"], n, seed, embed=embed)
        row = {"label": item["label"], "hypothesis": item["hypothesis"], **agg}
        results.append(row)
        with open(tsv, "a") as f:
            f.write("\t".join([item["label"], item["hypothesis"],
                               *[f"{_get(agg, c):.4f}" for c in METRIC_COLS]]) + "\n")
        print(f"  {item['label']:22s} velW={_get(agg, 'vel_w'):.4f} "
              f"jerkW={_get(agg, 'jerk_w'):.4f} posyW={_get(agg, 'posy_w'):.4f} "
              f"embCos={_get(agg, 'emb_cos'):.4f} FPD={_get(agg, 'emb_fpd'):.3f} "
              f"len={_get(agg, 'length_ratio'):.2f}")
    return results
