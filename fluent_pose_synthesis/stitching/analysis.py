"""Characterize how concatenated citation forms differ from fluent signing.

Runs over a sampled set of corpus sentences and reports, for the fluent
reference vs. the isolated dictionary forms:

* sentence / sign durations and the isolated-vs-fluent length ratio
* inter-sign gaps and co-articulation (temporal overlap of adjacent signs)
* hand velocity (tempo)

The findings drive which stitching techniques are worth trying.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fluent_pose_synthesis.stitching import dgs_corpus as dc
from fluent_pose_synthesis.stitching import dgs_types as dt
from fluent_pose_synthesis.stitching import harness as H
from fluent_pose_synthesis.stitching import metrics as M


def _pct(a):
    a = np.array(a, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {}
    return {
        "n": int(len(a)),
        "mean": round(float(np.mean(a)), 3),
        "median": round(float(np.median(a)), 3),
        "p10": round(float(np.percentile(a, 10)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
    }


def analyze(n: int, seed: int) -> dict:
    sents = H.build_eval_set(n=n, seed=seed)

    sent_dur_ms, glosses_per_sent = [], []
    fluent_sign_ms, isolated_sign_frames, fluent_sign_frames = [], [], []
    dur_ratio = []  # isolated_frames / fluent_frames per gloss
    inter_sign_gap_ms, overlap_ms = [], []
    fluent_speed, isolated_speed = [], []

    for s in sents:
        sent_dur_ms.append(s.duration_ms)
        lex = [g for g in s.glosses if g.is_lexical]
        glosses_per_sent.append(len(lex))

        prev_end = None
        for g in s.glosses:
            if prev_end is not None:
                gap = g.start_ms - prev_end
                if gap >= 0:
                    inter_sign_gap_ms.append(gap)
                else:
                    overlap_ms.append(-gap)
            prev_end = g.end_ms

        for g in lex:
            iso = dt.get_pose(g.gloss)
            if iso is None or iso.body.data.shape[0] == 0:
                continue
            fluent_ms = g.duration_ms
            fluent_f = max(1, round(fluent_ms / 1000 * 50))
            iso_f = iso.body.data.shape[0]
            fluent_sign_ms.append(fluent_ms)
            fluent_sign_frames.append(fluent_f)
            isolated_sign_frames.append(iso_f)
            dur_ratio.append(iso_f / fluent_f)
            isolated_speed.append(M.hand_speed(M.normalize(iso)))
            try:
                fl = dc.gloss_pose(s, g)
                if fl.body.data.shape[0] >= 2:
                    fluent_speed.append(M.hand_speed(M.normalize(fl)))
            except Exception:
                pass

    report = {
        "eval_set": {"n_sentences": len(sents), "seed": seed},
        "sentence_duration_ms": _pct(sent_dur_ms),
        "lexical_glosses_per_sentence": _pct(glosses_per_sent),
        "fluent_sign_duration_ms": _pct(fluent_sign_ms),
        "fluent_sign_frames": _pct(fluent_sign_frames),
        "isolated_sign_frames": _pct(isolated_sign_frames),
        "isolated_over_fluent_frame_ratio": _pct(dur_ratio),
        "inter_sign_gap_ms": _pct(inter_sign_gap_ms),
        "adjacent_sign_overlap_ms (co-articulation)": _pct(overlap_ms),
        "fluent_hand_speed_norm_per_frame": _pct(fluent_speed),
        "isolated_hand_speed_norm_per_frame": _pct(isolated_speed),
        "counts": {
            "overlapping_adjacent_pairs": len(overlap_ms),
            "gapped_adjacent_pairs": len(inter_sign_gap_ms),
        },
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="autoresearch/analysis.json")
    args = p.parse_args()
    report = analyze(args.n, args.seed)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
