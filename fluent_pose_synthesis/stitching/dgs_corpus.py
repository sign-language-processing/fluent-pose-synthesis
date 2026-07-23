"""Read the Public DGS Corpus locally: EAF annotations + holistic poses.

For each document we expose, per signer, the list of translated sentences and
the ordered sequence of glosses that make up each sentence, plus helpers to
slice the corresponding holistic-pose frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pose_format import Pose

from fluent_pose_synthesis.stitching.config import DGS_CORPUS_DIR, EVAL_CACHE
from fluent_pose_synthesis.stitching.pose_lookup import load_document_pose, load_pose

# Signer -> (video filename, EAF tier suffix)
SIGNERS = {"a": ("video_a.mp4", "A"), "b": ("video_b.mp4", "B")}


@dataclass
class Gloss:
    start_ms: int
    end_ms: int
    gloss: str  # German gloss (maps to DGS Types)
    english: str
    hand: str  # "r" or "l"

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def is_lexical(self) -> bool:
        """$-prefixed glosses are gestures/pointing, not dictionary lexemes."""
        return bool(self.gloss) and not self.gloss.startswith("$")


@dataclass
class Sentence:
    doc_id: str
    signer: str
    start_ms: int
    end_ms: int
    english: str
    german: str
    glosses: list[Gloss] = field(default_factory=list)
    video_path: Optional[Path] = None

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def key(self) -> str:
        return f"{self.doc_id}_{self.signer}_{self.start_ms}"


def list_documents() -> list[str]:
    """Document ids that have an EAF and at least one signer video."""
    docs = []
    for d in sorted((DGS_CORPUS_DIR / "videos").iterdir()):
        if not d.is_dir() or not (d / "data.eaf").exists():
            continue
        if any((d / v).exists() for v, _ in SIGNERS.values()):
            docs.append(d.name)
    return docs


def _merge_hand_glosses(right: list[Gloss], left: list[Gloss], overlap: float = 0.5) -> list[Gloss]:
    """Merge right/left tiers into one temporally-ordered lexical sequence.

    Two-handed signs are annotated on both tiers with (near) identical timing;
    we keep the dominant (right) copy and drop the left duplicate.  Left-only
    signs (no overlapping right gloss) are inserted in time order.
    """
    merged = list(right)
    for lg in left:
        dup = False
        for rg in right:
            inter = min(lg.end_ms, rg.end_ms) - max(lg.start_ms, rg.start_ms)
            if inter <= 0:
                continue
            shorter = max(1, min(lg.duration_ms, rg.duration_ms))
            if inter / shorter >= overlap:
                dup = True
                break
        if not dup:
            merged.append(lg)
    merged.sort(key=lambda g: g.start_ms)
    return merged


@lru_cache(maxsize=64)
def load_document_sentences(doc_id: str) -> tuple[Sentence, ...]:
    """All sentences (both signers) for a document, each with its gloss list."""
    import pympi

    doc_dir = DGS_CORPUS_DIR / "videos" / doc_id
    eaf = pympi.Elan.Eaf(str(doc_dir / "data.eaf"))
    tiers = set(eaf.get_tier_names())

    sentences: list[Sentence] = []
    for signer, (video, suffix) in SIGNERS.items():
        video_path = doc_dir / video
        if not video_path.exists():
            continue
        trans_tier = f"Translation_into_English_{suffix}"
        de_tier = f"Deutsche_Übersetzung_{suffix}"
        if trans_tier not in tiers:
            continue

        # Ordered gloss sequence from the two hand tiers.
        def _hand(tier_name: str, hand: str) -> list[Gloss]:
            if tier_name not in tiers:
                return []
            out = []
            for ann in eaf.get_annotation_data_for_tier(tier_name):
                start, end = int(ann[0]), int(ann[1])
                english = ann[2] if len(ann) > 2 else ""
                german = ann[3] if len(ann) > 3 else english
                out.append(Gloss(start, end, german or "", english or "", hand))
            return out

        right = _hand(f"Lexeme_Sign_r_{suffix}", "r")
        left = _hand(f"Lexeme_Sign_l_{suffix}", "l")
        all_glosses = _merge_hand_glosses(right, left)

        de_map = {int(a[0]): a[2] for a in eaf.get_annotation_data_for_tier(de_tier)} if de_tier in tiers else {}
        for ann in eaf.get_annotation_data_for_tier(trans_tier):
            start, end, english = int(ann[0]), int(ann[1]), ann[2] or ""
            glosses = [g for g in all_glosses if start <= g.start_ms < end]
            sentences.append(
                Sentence(
                    doc_id=doc_id,
                    signer=signer,
                    start_ms=start,
                    end_ms=end,
                    english=english,
                    german=de_map.get(start, ""),
                    glosses=glosses,
                    video_path=video_path,
                )
            )
    return tuple(sentences)


def slice_pose(pose: Pose, start_ms: float, end_ms: float) -> Pose:
    """Sub-pose covering [start_ms, end_ms) of a full-document holistic pose."""
    fps = pose.body.fps
    fa = max(0, round(start_ms / 1000 * fps))
    fb = min(len(pose.body.data), round(end_ms / 1000 * fps))
    return Pose(pose.header, pose.body[fa:fb])


def sentence_pose(sentence: Sentence) -> Pose:
    """The real, fluent holistic pose for a corpus sentence (the reference).

    Cached to disk as a small ``.pose``; the full ~300 MB document pose is
    loaded only on the first miss, which keeps repeated experiments off disk.
    """
    cache = EVAL_CACHE / f"ref_{sentence.key}.pose"
    if cache.exists():
        return load_pose(cache)
    doc_pose = load_document_pose(str(sentence.video_path))
    pose = slice_pose(doc_pose, sentence.start_ms, sentence.end_ms)
    with open(cache, "wb") as f:
        pose.write(f)
    return pose


def gloss_pose(sentence: Sentence, gloss: Gloss) -> Pose:
    """The fluent pose of a single gloss, cropped from its sentence video.

    Sliced from the (cached) sentence pose using offsets relative to the
    sentence start, so no document-pose reload is needed.
    """
    sent = sentence_pose(sentence)
    return slice_pose(sent, gloss.start_ms - sentence.start_ms, gloss.end_ms - sentence.start_ms)
