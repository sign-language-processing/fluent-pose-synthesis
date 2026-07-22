"""DGS Types dictionary: map a gloss to its isolated (citation-form) pose.

The dictionary index only stores video URLs, so the first time a gloss is
requested we download the clip, run MediaPipe holistic, and cache the pose.
Isolated forms are the disfluent building blocks a concatenative
text-to-pose system stitches together.
"""

from __future__ import annotations

import csv
import json
import re
import tempfile
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pose_format import Pose

from fluent_pose_synthesis.stitching.config import DGS_TYPES_DIR, TYPES_POSE_CACHE
from fluent_pose_synthesis.stitching.pose_lookup import extract_holistic

csv.field_size_limit(10**7)

_SUFFIX_RE = re.compile(r"[\^*]+$")


def strip_suffix(gloss: str) -> str:
    """Drop trailing corpus modifiers (``^``, ``*``) to reach the base type."""
    return _SUFFIX_RE.sub("", gloss)


@dataclass
class TypeEntry:
    type_id: str
    gloss: str
    video_url: str
    is_galex: bool


@lru_cache(maxsize=1)
def _load_index() -> dict[str, TypeEntry]:
    """gloss -> best TypeEntry. Prefer korpusdict (meinedgs) over GALEX."""
    index: dict[str, TypeEntry] = {}
    with open(DGS_TYPES_DIR / "index.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            glosses = json.loads(row["glosses"])
            views = json.loads(row["views"] or "[]")
            if not views:
                continue
            # Prefer the frontal view.
            front = next((v for v in views if v.get("name", "").lower() == "front"), views[0])
            video_url = front["video"]
            type_id = row["id"]
            is_galex = type_id.startswith("galex_")
            for gloss in glosses:
                entry = TypeEntry(type_id, gloss, video_url, is_galex)
                existing = index.get(gloss)
                # Keep the non-galex entry when both exist.
                if existing is None or (existing.is_galex and not is_galex):
                    index[gloss] = entry
    return index


def resolve(gloss: str) -> Optional[TypeEntry]:
    """Find the dictionary entry for a gloss (exact, then suffix-stripped)."""
    if not gloss or gloss.startswith("$"):
        return None
    index = _load_index()
    if gloss in index:
        return index[gloss]
    base = strip_suffix(gloss)
    return index.get(base)


def _cache_path(type_id: str) -> Path:
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", type_id)
    return TYPES_POSE_CACHE / f"{safe}.pose"


@lru_cache(maxsize=4096)
def get_pose(gloss: str) -> Optional[Pose]:
    """Isolated holistic pose for a gloss, or ``None`` if not in the dictionary.

    Downloads + extracts on first use, then reads from the on-disk cache.
    A sentinel ``.missing`` file records extraction failures so we don't retry
    a broken clip on every run.
    """
    entry = resolve(gloss)
    if entry is None:
        return None

    cache = _cache_path(entry.type_id)
    if cache.exists():
        with open(cache, "rb") as f:
            return Pose.read(f.read())
    if cache.with_suffix(".missing").exists():
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
            urllib.request.urlretrieve(entry.video_url, tmp.name)  # noqa: S310
            pose = extract_holistic(Path(tmp.name))
        with open(cache, "wb") as f:
            pose.write(f)
        return pose
    except Exception as exc:  # noqa: BLE001
        cache.with_suffix(".missing").write_text(str(exc))
        return None
