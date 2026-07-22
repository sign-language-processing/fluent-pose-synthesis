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
