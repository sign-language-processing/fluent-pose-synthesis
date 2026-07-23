"""SignCLIP embedding client + embedding-distance metrics.

Sends poses to the sign-clip container (github.com/sign/sign-language-assessment)
and returns 768-dim embeddings, so we can measure how *semantically* close a
stitched reconstruction is to the real sentence — a metric that captures
sign content, unlike keypoint DTW.

Run the service first:
    docker run --rm -p 8080:8080 -e PORT=8080 sign-clip
"""

from __future__ import annotations

import base64
import hashlib
import io
import os

import numpy as np
from pose_format import Pose

from fluent_pose_synthesis.stitching.config import CACHE_DIR

SIGNCLIP_URL = os.environ.get("SIGNCLIP_URL", "http://localhost:8080")
_EMB_CACHE_DIR = CACHE_DIR / "signclip_emb"
_EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _pose_bytes(pose: Pose) -> bytes:
    buf = io.BytesIO()
    pose.write(buf)
    return buf.getvalue()


def health(url: str = None) -> bool:
    import requests

    url = url or SIGNCLIP_URL
    try:
        return requests.get(f"{url}/health", timeout=5).status_code == 200
    except Exception:
        return False


def embed_pose(pose: Pose, url: str = None, model_name: str = "default") -> np.ndarray:
    """768-dim SignCLIP embedding for one pose, disk-cached by content hash."""
    import requests

    url = url or SIGNCLIP_URL
    raw = _pose_bytes(pose)
    key = hashlib.md5(raw + model_name.encode()).hexdigest()
    cache = _EMB_CACHE_DIR / f"{key}.npy"
    if cache.exists():
        return np.load(cache)

    payload = {"pose": [base64.b64encode(raw).decode("ascii")], "model_name": model_name}
    resp = requests.post(f"{url}/api/embed/pose", json=payload, timeout=120)
    resp.raise_for_status()
    emb = np.asarray(resp.json()["embeddings"], dtype=np.float64)[0]
    np.save(cache, emb)
    return emb


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(1.0 - np.dot(a, b) / denom) if denom else float("nan")


def embedding_distance(hypothesis: Pose, reference: Pose, url: str = None) -> float:
    """Cosine distance between the hypothesis and reference SignCLIP embeddings
    (lower = semantically closer). Returns NaN if the service is unreachable."""
    try:
        return cosine_distance(embed_pose(hypothesis, url), embed_pose(reference, url))
    except Exception:
        return float("nan")


def frechet_distance(hyp_embs: np.ndarray, ref_embs: np.ndarray, diagonal: bool = True) -> float:
    """Fréchet distance between two embedding sets (FID-style, distribution-level).

    With few samples relative to 768 dims, a full covariance is singular, so we
    default to a diagonal-covariance approximation.
    """
    mu_h, mu_r = hyp_embs.mean(0), ref_embs.mean(0)
    diff = float(np.sum((mu_h - mu_r) ** 2))
    if diagonal:
        vh, vr = hyp_embs.var(0), ref_embs.var(0)
        cov_term = float(np.sum(vh + vr - 2 * np.sqrt(np.clip(vh * vr, 0, None))))
        return diff + cov_term
    from scipy.linalg import sqrtm

    ch, cr = np.cov(hyp_embs, rowvar=False), np.cov(ref_embs, rowvar=False)
    covmean = sqrtm(ch @ cr).real
    return diff + float(np.trace(ch + cr - 2 * covmean))
