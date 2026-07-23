"""Evaluation harness: sample corpus sentences, reconstruct them by stitching
isolated dictionary forms, and score the result against the fluent reference.

The whole corpus is the training/eval set.  A stitching change is accepted only
if it improves the aggregate statistics on a fixed random sample versus the
baseline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from pose_format import Pose

from fluent_pose_synthesis.stitching import dgs_corpus as dc
from fluent_pose_synthesis.stitching import dgs_types as dt
from fluent_pose_synthesis.stitching import metrics as M

StitchFn = Callable[[list[Pose]], Pose]


@dataclass
class GlossSource:
    gloss: str
    pose: Pose
    source: str  # "types" | "sentence" (crop fallback)


def gloss_sources(sentence: dc.Sentence) -> list[GlossSource]:
    """Ordered building blocks for a sentence reconstruction.

    Lexical glosses present in DGS Types use the isolated citation form.
    Everything else (gestures, out-of-dictionary lexemes) is cropped straight
    from the original sentence video — we know the exact frames.
    """
    out: list[GlossSource] = []
    for g in sentence.glosses:
        pose = dt.get_pose(g.gloss) if g.is_lexical else None
        if pose is not None and pose.body.data.shape[0] > 0:
            out.append(GlossSource(g.gloss, pose, "types"))
        else:
            crop = dc.gloss_pose(sentence, g)
            if crop.body.data.shape[0] > 0:
                out.append(GlossSource(g.gloss, crop, "sentence"))
    return out


def coverage(sentence: dc.Sentence) -> float:
    """Fraction of lexical glosses resolvable in the dictionary (no download)."""
    lex = [g for g in sentence.glosses if g.is_lexical]
    if not lex:
        return 0.0
    return sum(dt.resolve(g.gloss) is not None for g in lex) / len(lex)


def build_eval_set(
    n: int = 40,
    seed: int = 0,
    min_lexical: int = 3,
    max_lexical: int = 15,
    min_ms: int = 1000,
    max_ms: int = 8000,
    min_coverage: float = 0.5,
) -> list[dc.Sentence]:
    """Deterministically sample eligible sentences spread across documents."""
    rng = random.Random(seed)
    docs = dc.list_documents()
    rng.shuffle(docs)

    picked: list[dc.Sentence] = []
    for doc in docs:
        try:
            sents = dc.load_document_sentences(doc)
        except Exception:
            continue
        eligible = [
            s for s in sents
            if min_lexical <= sum(g.is_lexical for g in s.glosses) <= max_lexical
            and min_ms <= s.duration_ms <= max_ms
            and coverage(s) >= min_coverage
        ]
        if eligible:
            picked.append(rng.choice(eligible))
        if len(picked) >= n:
            break
    # Group by video so the LRU document-pose cache stays warm.
    picked.sort(key=lambda s: str(s.video_path))
    return picked


def reconstruct(sentence: dc.Sentence, stitch_fn: StitchFn) -> Optional[Pose]:
    sources = gloss_sources(sentence)
    if not sources:
        return None
    return stitch_fn([s.pose for s in sources])


def evaluate_one(sentence: dc.Sentence, stitch_fn: StitchFn) -> Optional[dict]:
    try:
        hyp = reconstruct(sentence, stitch_fn)
        if hyp is None:
            return None
        ref = dc.sentence_pose(sentence)
        row = M.score(hyp, ref)
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "key": sentence.key}
    row["key"] = sentence.key
    row["n_glosses"] = len(sentence.glosses)
    row["coverage"] = coverage(sentence)
    return row


def evaluate(sentences: list[dc.Sentence], stitch_fn: StitchFn, verbose: bool = False) -> list[dict]:
    rows = []
    for i, s in enumerate(sentences):
        row = evaluate_one(s, stitch_fn)
        rows.append(row)
        if verbose and row is not None:
            if "error" in row:
                print(f"  [{i+1}/{len(sentences)}] {s.key} ERROR {row['error']}")
            else:
                print(f"  [{i+1}/{len(sentences)}] {s.key} dtwp={row['dtwp']:.1f} "
                      f"len={row['length_ratio']:.2f} cov={row['coverage']:.2f}")
    return rows
