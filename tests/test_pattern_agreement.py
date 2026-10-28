"""Tests for analysis.correlation.pattern_agreement (SI random re-pairing test)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from analysis.correlation.pattern_agreement import repairing_test

CONCEPTS = list("abcdefgh")
PAIRS = list(itertools.combinations(CONCEPTS, 2))


def test_identical_topic_specific_patterns_beat_random_repairing():
    rng = np.random.default_rng(1)
    values = {frozenset(p): v for p, v in zip(PAIRS, rng.uniform(0, 0.5, len(PAIRS)))}
    res = repairing_test(values, values, CONCEPTS, n_perm=500)
    assert res["diff"] == pytest.approx(0.0)
    assert res["corr"] == pytest.approx(1.0)
    assert res["diff_random"] > 0.05
    assert res["p_diff"] < 0.05 and res["p_corr"] < 0.05
    assert res["n_pairs"] == 28


def test_unrelated_patterns_are_no_better_than_random():
    rng = np.random.default_rng(2)
    first = {frozenset(p): v for p, v in zip(PAIRS, rng.uniform(0, 0.5, len(PAIRS)))}
    second = {frozenset(p): v for p, v in zip(PAIRS, rng.uniform(0, 0.5, len(PAIRS)))}
    res = repairing_test(first, second, CONCEPTS, n_perm=500)
    assert res["p_diff"] > 0.05
    assert res["diff"] == pytest.approx(res["diff_random"], abs=0.03)


def test_sign_is_ignored():
    values = {frozenset(p): 0.1 * (i % 5) for i, p in enumerate(PAIRS)}
    flipped = {k: -v for k, v in values.items()}
    assert repairing_test(values, flipped, CONCEPTS, n_perm=50)["diff"] == pytest.approx(0.0)
