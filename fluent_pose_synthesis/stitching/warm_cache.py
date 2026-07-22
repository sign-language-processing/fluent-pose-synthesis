"""Pre-extract every DGS Types pose needed for an eval set, in parallel.

Extraction (download + MediaPipe holistic) is the slow part; once cached, all
stitching experiments run fast.  Safe to re-run — already-cached glosses are
skipped.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from fluent_pose_synthesis.stitching import dgs_types as dt
from fluent_pose_synthesis.stitching import harness as H


def _extract(gloss: str) -> tuple[str, bool]:
    pose = dt.get_pose(gloss)
    return gloss, pose is not None


def collect_glosses(n: int, seed: int) -> list[str]:
    sents = H.build_eval_set(n=n, seed=seed)
    glosses = set()
    for s in sents:
        for g in s.glosses:
            if g.is_lexical and dt.resolve(g.gloss) is not None:
                glosses.add(g.gloss)
    return sorted(glosses)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 6

    glosses = collect_glosses(n, seed)
    print(f"eval set n={n} seed={seed}: {len(glosses)} unique dictionary glosses to ensure cached")
    ok = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_extract, g): g for g in glosses}
        for i, fut in enumerate(as_completed(futures), 1):
            g, success = fut.result()
            ok += success
            if i % 10 == 0 or i == len(glosses):
                print(f"  {i}/{len(glosses)} done ({ok} ok)")
    print(f"cache warm complete: {ok}/{len(glosses)} extracted")


if __name__ == "__main__":
    main()
