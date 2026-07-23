"""Multi-objective analysis of an iteration results.tsv.

DTW-family metrics each have known biases (raw favors short; clean/norm favor
smooth/long), so we rank by several objectives and surface the Pareto frontier
of shape vs. length, plus the distribution-match metrics (trustworthy quality).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load(tsv: Path) -> list[dict]:
    rows = []
    with open(tsv) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            for k, v in list(r.items()):
                if k not in ("label", "hypothesis"):
                    try:
                        r[k] = float(v)
                    except (ValueError, TypeError):
                        r[k] = float("nan")
            rows.append(r)
    # de-dup by label keeping the last occurrence
    seen = {}
    for r in rows:
        seen[r["label"]] = r
    return list(seen.values())


def top(rows, key, n=8, reverse=False, length_band=None):
    cand = rows
    if length_band:
        lo, hi = length_band
        cand = [r for r in rows if lo <= r.get("length_ratio_mean", 0) <= hi]
    cand = [r for r in cand if r.get(f"{key}_mean") == r.get(f"{key}_mean")]  # drop nan
    return sorted(cand, key=lambda r: r[f"{key}_mean"], reverse=reverse)[:n]


def show(title, rows, key):
    print(f"\n== {title} ==")
    print(f"{'label':24s} {'clean':>6s} {'dtwpM':>6s} {'len':>5s} {'velW':>7s} {'stillErr':>8s}")
    for r in rows:
        print(f"{r['label']:24s} {r.get('dtwp_clean_mean', float('nan')):6.3f} "
              f"{r.get('dtwp_matched_mean', float('nan')):6.2f} "
              f"{r.get('length_ratio_mean', float('nan')):5.2f} "
              f"{r.get('vel_w_mean', float('nan')):7.4f} "
              f"{r.get('still_frac_mean', float('nan')):8.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tsv", required=True)
    args = p.parse_args()
    rows = load(Path(args.tsv))
    print(f"loaded {len(rows)} unique configs")
    show("Best dtwp_matched @ natural length [0.9,1.5]", top(rows, "dtwp_matched", length_band=(0.9, 1.5)), "dtwp_matched")
    show("Best velocity-distribution match (velW) @ len [0.9,1.6]", top(rows, "vel_w", length_band=(0.9, 1.6)), "vel_w")
    show("Lowest still-frac (ref≈0.14) @ len [0.9,1.6]", top(rows, "still_frac", length_band=(0.9, 1.6)), "still_frac")
    show("Best clean shape @ len [0.9,1.6]", top(rows, "dtwp_clean", length_band=(0.9, 1.6)), "dtwp_clean")


if __name__ == "__main__":
    main()
