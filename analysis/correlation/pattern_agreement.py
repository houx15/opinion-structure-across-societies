"""Is the agreement between two societies' topic-pair correlations topic-specific?

U.S. and British sources share eight topic concepts (stats Part 12, SI).
For each medium we compare the two societies' |r| over the 28 concept pairs:
mean absolute difference and the correlation of the two patterns. A random
re-pairing relabels one society's concepts by a random permutation, which
keeps both sets of values but breaks the correspondence of topics. If the
matched agreement is no better than under random re-pairing, the two sources
are alike only in level (e.g. everything compressed to a common value), not
in which topics go together.
"""

from __future__ import annotations

import itertools
from typing import Dict, FrozenSet, Hashable, Mapping, Sequence

import numpy as np

PairValues = Mapping[FrozenSet[Hashable], float]


def _vector(values: PairValues, pairs, relabel=None) -> np.ndarray:
    relabel = relabel or {}
    return np.array([abs(values[frozenset((relabel.get(a, a), relabel.get(b, b)))]) for a, b in pairs])


def repairing_test(
    first: PairValues, second: PairValues, concepts: Sequence[Hashable],
    n_perm: int = 5000, seed: int = 0,
) -> Dict[str, float]:
    """Matched vs randomly re-paired agreement of |r| between two societies.

    ``first`` and ``second`` map frozenset({concept_a, concept_b}) to r (the
    sign is dropped). Returns the matched mean |difference| and pattern
    correlation, their means under ``n_perm`` random permutations of the
    second society's concepts, and the shares of permutations at least as
    close as the matched pairing (one-sided permutation p-values).
    """
    concepts = list(concepts)
    pairs = list(itertools.combinations(concepts, 2))
    x = _vector(first, pairs)
    y = _vector(second, pairs)
    diff = float(np.mean(np.abs(x - y)))
    corr = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    diffs, corrs = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        relabel = dict(zip(concepts, rng.permutation(concepts)))
        yp = _vector(second, pairs, relabel)
        diffs[i] = np.mean(np.abs(x - yp))
        corrs[i] = np.corrcoef(x, yp)[0, 1]
    return {
        "n_concepts": len(concepts), "n_pairs": len(pairs),
        "diff": diff, "diff_random": float(diffs.mean()), "p_diff": float(np.mean(diffs <= diff)),
        "corr": corr, "corr_random": float(corrs.mean()), "p_corr": float(np.mean(corrs >= corr)),
    }
