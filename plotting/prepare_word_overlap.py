"""Precompute the data for plotting.figures Fig 2 Row 2 (word-overlap panel).

The plotting code re-runs the same calculation every time it draws Fig 2: pick
US Twitter's lowest- and highest-similarity topic pair, load top-K TF-IDF
words for each topic, look up GPT embeddings, run a greedy pairwise
cosine-similarity match. Nothing here depends on the task being plotted —
it's the same input on every call — so the result is cacheable.

This script writes one small JSON artifact (`data/embedding/word_overlap_rows.json`)
with the precomputed rows. ``plotting.figures._word_overlap_panel`` reads that
artifact when it's present and falls back to a live compute when it isn't,
so plotting still works on a fresh checkout.

Usage (from the repository root):
    python -m plotting.prepare_word_overlap compute
    python -m plotting.prepare_word_overlap compute --top_k 15
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import fire

from plotting import figures as v4


DEFAULT_OUTPUT_PATH = v4.EMBEDDING_DIR / "word_overlap_rows.json"


def _build_payload(top_k: int) -> dict:
    """Return the JSON payload to write to disk."""
    low, high = v4._select_us_twitter_extremes()
    embeddings = v4._load_word_embeddings()
    pairs = {}
    for label, (t1, t2, pair_sim) in (("low", low), ("high", high)):
        wa = v4._load_twitter_top_words_with_weights(t1, k=top_k)
        wb = v4._load_twitter_top_words_with_weights(t2, k=top_k)
        rows = v4._compute_word_overlap_rows(wa, wb, embeddings=embeddings)
        pairs[label] = {
            "topic_a": t1,
            "topic_b": t2,
            "topic_similarity": float(pair_sim),
            "words_a": [{"word": w, "tf_idf": s} for w, s in wa],
            "words_b": [{"word": w, "tf_idf": s} for w, s in wb],
            "rows": [asdict(r) for r in rows],
        }
    return {
        "source": "twitter",
        "embedding_type": "gpt",
        "top_k": top_k,
        "pairs": pairs,
    }


def compute(
    output_path: str = str(DEFAULT_OUTPUT_PATH),
    top_k: int = v4._WORD_OVERLAP_TOP_K,
) -> str:
    """Compute the word-overlap rows and save to ``output_path`` (JSON).

    Returns the output path (so the CLI prints it). Pass --top_k to override
    the default top-K (currently {default_k}); the default matches
    ``semantic_similarity.process_embedding``.
    """
    payload = _build_payload(top_k=top_k)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    n_rows_low = len(payload["pairs"]["low"]["rows"])
    n_rows_high = len(payload["pairs"]["high"]["rows"])
    print(
        f"[word_overlap] wrote {out}  top_k={top_k}  "
        f"low={payload['pairs']['low']['topic_a']}×{payload['pairs']['low']['topic_b']} "
        f"({n_rows_low} rows)  "
        f"high={payload['pairs']['high']['topic_a']}×{payload['pairs']['high']['topic_b']} "
        f"({n_rows_high} rows)"
    )
    return str(out)


compute.__doc__ = (compute.__doc__ or "").format(default_k=v4._WORD_OVERLAP_TOP_K)


if __name__ == "__main__":
    fire.Fire({"compute": compute})
