# DGS Corpus vs. concatenative reconstruction — analysis & stitching experiments

All numbers are over deterministic random samples of DGS Corpus sentences
(one sentence per document, ≥3 lexical glosses, ≥50 % dictionary coverage).
`seed 0` and `seed 1` are disjoint 40-sentence samples.

## 1. How the two signals differ

Corpus (fluent, continuous signing) vs. DGS Types (isolated citation forms),
n=40 sentences, 189 signs:

| Property | Fluent (corpus) | Isolated (DGS Types) |
|---|---|---|
| Sign duration | median **10 frames** (200 ms) | median **116 frames** (2.3 s) |
| Isolated ÷ fluent frame ratio | — | **mean 14.7×, median 11.7×** |
| Hand speed (norm/frame) | 0.023 | 0.021 |
| Inter-sign gap | median 180 ms | (n/a — clips are isolated) |
| Sentence duration | median 3.1 s, 5.5 lexical signs | — |

**The dominant difference is duration, not velocity.** Citation forms are
~12–15× longer than the same sign in fluent context, but move at essentially
the same per-frame speed — the extra length is preparation, holds and
retraction, plus fuller (un-reduced, un-coarticulated) movement.

## 2. What the baseline (spoken-to-signed) produces

The baseline `concatenate_poses` (reduce-holistic → shoulder-normalize →
hand-raise trim → closest-frame join + 0.20 s padding + Savitzky–Golay) already
removes most citation-form holds, but the stitched sentence is still
**2.6× too long** on average (seed 0: len_ratio 2.60; seed 1: 2.50).

## 3. Metric note — DTWp is length-biased

pose-evaluation's DTWp (dtaidistance) returns a **cumulative** warping-path
distance, so it is mechanically lower for **shorter** hypotheses. Reported here:

* **dtwp_raw** — pose-evaluation DTWp as-is (length-biased, keep for reference).
* **dtwp_norm** — the same distance divided by the warping-path length
  (mean per-aligned-step distance): a length-fair measure of trajectory *shape*.
* **len_ratio** — hypothesis frames ÷ reference frames (timing fidelity, →1.0).

Raw DTWp alone is misleading: `fps_12` posts the "best" raw DTWp (48.6) purely
because it is 0.74× length — its shape (`dtwp_norm` 1.12) is the *worst*.

## 4. Experiment sweep (seed 0)

Ranked by shape-fair `dtwp_norm`:

| config | dtwp_norm | dtwp_raw | len_ratio | note |
|---|---|---|---|---|
| baseline | **0.542** | 58.60 | 2.60 | best shape, worst length |
| pad_0.05 | 0.587 | 56.85 | 2.32 | drop inter-sign padding 0.20→0.05 |
| pad_0.00 | 0.599 | 56.40 | 2.25 | no padding |
| seg_trim | 0.697 | 54.43 | 1.89 | **segmentation-model trimming** |
| seg_pad05 | 0.770 | 52.15 | 1.61 | seg trim + pad 0.05 |
| seg_pad00 | 0.793 | 51.53 | 1.54 | seg trim + no padding |
| cap_30 | 0.897 | 54.59 | 1.23 | cap only signs >30 frames |
| cap_25 | 0.969 | 53.66 | 1.08 | cap only signs >25 frames |
| fps_20 | 0.992 | 52.77 | 1.03 | resample every sign to 20 frames |
| fps_12 | 1.122 | 48.57 | 0.74 | over-compressed |

Held-out **seed 1** reproduces the ordering and values (baseline 0.560/2.50,
seg_pad00 0.779/1.53, cap_25 0.861/1.08, fps_20 0.889/1.04).

## 5. Conclusion

There is a **Pareto trade-off between timing fidelity and trajectory shape**:

* No configuration beats the baseline on *both* axes.
* The baseline already achieves the best per-frame **shape**; DTW warps away
  its excess length, so shape can't be improved much by re-timing.
* The baseline's real defect is **over-length (2.6×)**. This *is* fixable:
  - **Segmentation-model trimming** cuts citation-form holds far better than the
    hand-raise heuristic, taking length 2.6×→~1.5–1.9× at modest shape cost —
    the best point on the frontier for "natural but faithful".
  - **Per-sign duration capping / resampling** reaches fluent length (~1.0×) but
    at a real shape cost, because citation *movement* (not just holds) is fuller
    than fluent movement.

**The residual gap — matching fluent tempo without distorting shape — is
exactly what a learned model (the original fluent-pose-synthesis idea) is needed
for.** Pure stitching gets timing most of the way there; it cannot invent the
coarticulation that makes fluent signing both short and well-formed.

**Recommended default:** `seg_pad00` (segmentation trim + no padding) — roughly
halves the over-length with the least shape cost of the natural-length configs,
and degrades gracefully to the hand-raise heuristic when the segmentation model
is unavailable. Use `cap_25` when tight tempo matching matters more than shape.

---

# Part 2 — Deep study: can we actually move DTWp? (100+ iterations)

A second round targeted DTWp *itself* (the hard metric), diagnosing error
sources statistically and testing >100 configurations (`docs/iteration_results.tsv`),
drawing on **Sign Stitching** (Walsh et al., BMVC 2024) and the "detach the
hands / minimize inter-sign movement" idea.

## Better metrics for a fair fight

Every DTW-family score has a bias, so we track three plus distribution stats:

* `dtwp` — pose-evaluation DTWp as-is (cumulative → favors *short*).
* `dtwp_matched` — DTWp after resampling the hypothesis to the reference length
  (isolates shape; the main objective).
* `dtwp_clean` — path-normalized joint-hand DTW with masked keypoints
  *interpolated* instead of filled with 10.0 (artifact-free; but favors *smooth*).
* `vel_w`, `still_frac`, position offsets — mask-aware **distribution** metrics
  (velocity/hold/position), the most trustworthy quality signal.

## Diagnosis — where the DTWp error actually is

* **~half of raw DTWp is a metric artifact.** pose-evaluation fills undetected
  keypoints with 10.0; the reference has ~2 % masked hand frames (a resting hand),
  and each becomes a ~10-unit spike. The reference's apparent huge spread
  (velocity mean 0.40, position std 1.4) is entirely these fills — mask-aware,
  the medians nearly match (hyp 0.017, ref 0.026).
* **Position dominates over handshape**: wrist-trajectory DTW (28.9) ≈ finger DTW
  (30.2); since finger coords are absolute, most of the distance is *where the
  hand is*, not its configuration.
* **Depth (z) is uninformative** — both signals have z-std ≈ 0.001 after
  normalization (MediaPipe depth); the earlier "z problem" was the 10.0 fill.
* **Real, fixable distribution gaps**: the stitch is jitterier (velocity std
  0.072 vs 0.025), holds too much (still-frac 0.37 vs 0.14), and sits ~0.16 lower
  in *y* than the reference.

## Techniques tested (each measured vs. the seg anchor)

Segmentation-trim + no-padding (`seg_pad00`, dtwp_matched **46.7**, len 1.41) is
the anchor. Over 100+ configs:

| technique | effect on dtwp_matched | effect on distribution |
|---|---|---|
| Butterworth low-pass (Sign Stitching) | 9 Hz: **46.5** (best); 4 Hz: worse | **velW −12–36 %**, fewer holds |
| velocity-matched / direct transitions | worse (adds length) | slightly smoother |
| co-articulation (detach & pull signs together) | neutral→worse | mild velW help |
| per-sign resample / duration cap | worse (distorts shape) | matches length & hold-fraction |
| hand y-shift (distribution align) | −0.1 % (negligible) | small position match |
| motion-threshold crop | ≈ hand-raise | — |

## Conclusion of the deep study

**DTWp shape has a hard floor at ~the segmentation-trim config — no stitching
geometry beat it by more than ~0.4 % across 100+ tries**, on two disjoint
samples. Raw-DTW "wins" are always length or smoothness artifacts. This is
strong evidence that the citation-vs-fluent gap (coarticulation, movement
reduction) is **not recoverable by concatenation-level tricks** — it needs a
learned model. What stitching *can* robustly deliver:

1. **Segmentation trimming** — lower dtwp_matched and ~half the length vs. the
   spoken-to-signed baseline (both seeds).
2. **Butterworth low-pass (~9 Hz)** — smoother, more natural velocity
   distribution and fewer spurious holds, at **zero** dtwp cost.

**Recommended default (`fluent` preset): segmentation trim + Butterworth 9 Hz.**
Held-out seed 1: dtwp_matched 50.6 (baseline) → 49.1 (−3 %), length 2.50 → 1.53,
velW 0.024 → 0.022. `fluent_short` adds a duration cap for ~natural length when
tempo fidelity outweighs shape.
