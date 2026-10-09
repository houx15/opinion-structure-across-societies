"""Compare recomputed PR / eRank / srank with an earlier results folder.

    python scripts/compare_dimension_results.py /path/to/old/dimension_data
    python scripts/compare_dimension_results.py OLD --new data/dimension/results/cov --tol 1e-9

For every <stem>-none-summary.json present in both folders, prints the
largest absolute difference per metric. Stems only in one folder are listed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.paths import dimension_results_dir  # noqa: E402

SUFFIX = "-none-summary.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("old", type=Path, help="earlier results folder")
    ap.add_argument("--new", type=Path, default=dimension_results_dir("cov"))
    # Results computed before the ridge (dimensions.RIDGE = 1e-6) was enabled
    # differ from current ones by ~1e-6; use e.g. --tol 1e-4 for those.
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args()

    stems = lambda d: {f.name[: -len(SUFFIX)] for f in d.glob(f"*{SUFFIX}")}
    old, new = stems(args.old), stems(args.new)
    print(f"{'stem':22} {'metric':6} {'old mean':>10} {'new mean':>10} {'max |diff|':>11}")
    for stem in sorted(old & new):
        a = json.loads((args.old / f"{stem}{SUFFIX}").read_text())
        b = json.loads((args.new / f"{stem}{SUFFIX}").read_text())
        for metric in ("PR", "eRank", "srank"):
            x, y = a.get(metric, []), b.get(metric, [])
            if len(x) != len(y):
                print(f"{stem:22} {metric:6} years differ: {len(x)} vs {len(y)}")
                continue
            diff = max((abs(p - q) for p, q in zip(x, y)), default=0.0)
            flag = "" if diff <= args.tol else "  <-- differs"
            print(f"{stem:22} {metric:6} {sum(x) / len(x):10.4f} {sum(y) / len(y):10.4f} {diff:11.2e}{flag}")
    for label, only in (("only in old", old - new), ("only in new", new - old)):
        if only:
            print(f"{label}: {', '.join(sorted(only))}")


if __name__ == "__main__":
    main()
