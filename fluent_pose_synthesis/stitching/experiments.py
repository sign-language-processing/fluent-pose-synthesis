"""A/B-test stitching configs against the baseline on a fixed corpus sample.

Each config is one change (or a combination) on top of the spoken-to-signed
baseline.  We evaluate every config on the *same* sampled sentences and report
DTWp and length statistics, so an improvement is measured, not assumed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

from fluent_pose_synthesis.stitching import harness as H
from fluent_pose_synthesis.stitching import metrics as M
from fluent_pose_synthesis.stitching.concatenate import StitchConfig, concatenate_poses

# --- Candidate configs ---------------------------------------------------
CONFIGS: dict[str, StitchConfig] = {
    "baseline": StitchConfig(),
    # Inter-sign padding sweep (baseline pads 0.20s between every sign)
    "pad_0.10": StitchConfig(padding=0.10),
    "pad_0.05": StitchConfig(padding=0.05),
    "pad_0.00": StitchConfig(padding=0.00),
    # Global tempo compression of citation forms
    "speed_1.5": StitchConfig(speed=1.5),
    "speed_2.0": StitchConfig(speed=2.0),
    "speed_2.5": StitchConfig(speed=2.5),
    "speed_3.0": StitchConfig(speed=3.0),
    # Resample every sign to a fixed length
    "fps_20": StitchConfig(frames_per_sign=20),
    "fps_15": StitchConfig(frames_per_sign=15),
    "fps_12": StitchConfig(frames_per_sign=12),
    # Complexity-scaled target length
    "fps_18_cx": StitchConfig(frames_per_sign=18, complexity_scaling=True),
    # Segmentation-based trimming (cuts citation-form holds)
    "seg_trim": StitchConfig(trim_method="segmentation"),
    "seg_pad05": StitchConfig(trim_method="segmentation", padding=0.05),
    "seg_pad00": StitchConfig(trim_method="segmentation", padding=0.00),
    "seg_speed15": StitchConfig(trim_method="segmentation", speed=1.5, padding=0.05),
    "seg_fps_18cx": StitchConfig(trim_method="segmentation", frames_per_sign=18,
                                 complexity_scaling=True, padding=0.05),
    # Combined best-guess: segmentation trim + light padding + mild compression
    "combo": StitchConfig(trim_method="segmentation", speed=1.3, padding=0.05),
    # Per-sign duration cap: compress only over-long signs, leave short ones alone
    "cap_30": StitchConfig(max_sign_frames=30),
    "cap_25": StitchConfig(max_sign_frames=25),
    "seg_cap30": StitchConfig(trim_method="segmentation", max_sign_frames=30, padding=0.05),
    "seg_cap25": StitchConfig(trim_method="segmentation", max_sign_frames=25, padding=0.05),
    "seg_cap20": StitchConfig(trim_method="segmentation", max_sign_frames=20, padding=0.05),
}


def stitch_fn(config: StitchConfig):
    def fn(poses):
        with contextlib.redirect_stdout(io.StringIO()):
            return concatenate_poses(poses, config)
    return fn


def run(config_names: list[str], n: int, seed: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    sents = H.build_eval_set(n=n, seed=seed)
    print(f"eval set: {len(sents)} sentences (n={n}, seed={seed})")

    results = {}
    for name in config_names:
        cfg = CONFIGS[name]
        t = time.time()
        rows = H.evaluate(sents, stitch_fn(cfg))
        valid = [r for r in rows if r and "error" not in r]
        agg = M.aggregate(valid)
        agg["config"] = name
        agg["seconds"] = round(time.time() - t, 1)
        agg["errors"] = len([r for r in rows if r and "error" in r])
        results[name] = {"agg": agg, "rows": rows}
        _dump_rows(out_dir / f"rows_{name}_n{n}_s{seed}.tsv", rows)
        print(f"  {name:14s} dtwp={agg.get('dtwp_mean', float('nan')):7.2f} "
              f"dtwpN={agg.get('dtwp_norm_mean', float('nan')):6.3f} "
              f"len={agg.get('length_ratio_mean', float('nan')):5.2f} "
              f"absΔ={agg.get('length_abs_err_mean', float('nan')):6.1f} "
              f"({agg['seconds']}s, {agg['errors']} err)")

    _print_comparison(results, out_dir, n, seed)
    return results


def _dump_rows(path: Path, rows: list[dict]) -> None:
    cols = ["key", "dtwp", "dtwp_norm", "length_ratio", "length_abs_err", "hyp_frames", "ref_frames", "coverage", "n_glosses", "error"]
    with open(path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            if not r:
                continue
            f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


def _print_comparison(results: dict, out_dir: Path, n: int, seed: int) -> None:
    base = results.get("baseline")
    lines = ["", "=== SUMMARY (ranked by dtwp_norm; shape-fair, length-unbiased) ==="]
    lines.append(f"{'config':16s} {'dtwp_norm':>10s} {'Δ%':>7s} {'dtwp_raw':>9s} {'len_ratio':>10s} {'len_absΔ':>9s}")
    base_norm = base["agg"]["dtwp_norm_mean"] if base else None
    ranked = sorted(results.items(), key=lambda kv: kv[1]["agg"].get("dtwp_norm_mean", 1e9))
    for name, res in ranked:
        a = res["agg"]
        d = a.get("dtwp_norm_mean", float("nan"))
        pct = 100 * (d - base_norm) / base_norm if base_norm else float("nan")
        lines.append(f"{name:16s} {d:10.3f} {pct:+7.1f} {a.get('dtwp_mean', float('nan')):9.2f} "
                     f"{a.get('length_ratio_mean', float('nan')):10.2f} "
                     f"{a.get('length_abs_err_mean', float('nan')):9.1f}")
    text = "\n".join(lines)
    print(text)
    summary = {name: res["agg"] for name, res in results.items()}
    (out_dir / f"summary_n{n}_s{seed}.json").write_text(json.dumps(summary, indent=2))
    (out_dir / f"summary_n{n}_s{seed}.txt").write_text(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--configs", nargs="*", default=list(CONFIGS.keys()))
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="autoresearch/latest")
    args = p.parse_args()
    run(args.configs, args.n, args.seed, Path(args.out))


if __name__ == "__main__":
    main()
