"""Copy precomputed inputs from the original working repository into data/.

The original project kept its inputs in ``data/``, ``tf_idf/``,
``embedding/``, ``dimension_data/`` and ``dimension/data/`` of one working
folder. This script copies them, unchanged, into the layout of this
repository (see data/README.md):

    python scripts/import_legacy_data.py /path/to/opinion_correlation
    python scripts/import_legacy_data.py /path/to/opinion_correlation --csr_root /path/to/dimension/data
    python scripts/import_legacy_data.py OLD --dry_run   # list what would be copied

Nothing is converted; files keep their names and formats. Not copied:
* dimension results of the stems ``evs`` / ``evs_media``: the original
  pipeline computed them from evs_resample2.dta / evs_media_resample2.dta
  (identical to the *_resample2 results); rerun the survey dimension step to
  get the no-replacement samples.
* survey csr matrices (cheap to rebuild from data/survey/*.dta).
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import paths  # noqa: E402

SURVEYS = [
    "anes", "anes_media", "wvs", "wvs_media",
    "evs", "evs_media", "evs_resample2", "evs_media_resample2",
    "en_evs", "en_evs_media", "eu_nen_evs",
]
SOCIAL = ["twitter", "eutwitter", "entwitter", "eu_nentwitter", "weibo"]
TWITTER_MODES = ["us", "eu", "en", "eu_nen"]
# dimension results computed from the wrong input (see module docstring)
SKIP_DIMENSION_STEMS = ("evs-", "evs_media-")


class Importer:
    def __init__(self, dry_run: bool):
        self.dry_run = dry_run
        self.copied, self.missing = 0, []

    def put(self, src: Path, dst: Path, required: bool = True) -> None:
        if not src.exists():
            if required:
                self.missing.append(str(src))
            return
        print(f"  {src}  ->  {paths.display(dst)}")
        self.copied += 1
        if self.dry_run:
            return
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_dir():
            shutil.rmtree(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("legacy_root", type=Path, help="original working folder (opinion_correlation)")
    ap.add_argument("--csr_root", type=Path, default=None,
                    help="folder holding the social-media csr matrices "
                         "(twitter/<mode>/*.csr.npz, weibo/*.csr.npz); default <legacy_root>/dimension/data")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    L = args.legacy_root.resolve()
    imp = Importer(args.dry_run)

    print("[survey microdata]")
    for s in SURVEYS:
        imp.put(L / "data" / f"{s}.dta", paths.SURVEY_DIR / f"{s}.dta")

    print("[reference tables]")
    imp.put(L / "data" / "survey_codebook.xlsx", paths.REFERENCE_DIR / "survey_codebook.xlsx")
    imp.put(L / "data" / "new_topic_opinion.csv", paths.REFERENCE_DIR / "weibo_topics.csv")
    imp.put(L / "data" / "twitter-opinion-config-detail.csv",
            paths.REFERENCE_DIR / "twitter_topics.csv", required=False)

    print("[cleaned opinions]")
    for s in SURVEYS:
        imp.put(L / "data" / f"individual_opinion_{s}.parquet",
                paths.OPINION_DIR / f"individual_opinion_{s}.parquet")
    for s in SOCIAL:
        imp.put(L / "data" / f"user_opinion_{s}_lgbt_env.parquet",
                paths.OPINION_DIR / f"user_opinion_{s}_lgbt_env.parquet")

    print("[pairwise correlations]")
    for s in SURVEYS + SOCIAL:
        imp.put(L / "data" / f"network_analysis_{s}.csv",
                paths.CORRELATION_DIR / f"network_analysis_{s}.csv")

    print("[TF-IDF keyword tables]")
    for s in SOCIAL:
        imp.put(L / "tf_idf" / f"tf_idf_{s}", paths.TF_IDF_DIR / f"tf_idf_{s}")
    for f in sorted((L / "tf_idf").glob("survey_*_topic_*.csv")):
        imp.put(f, paths.TF_IDF_DIR / f.name)

    print("[embeddings and topic similarities]")
    E = L / "embedding"
    imp.put(E / "cached_embedding.pkl", paths.EMBEDDING_DIR / "cached_embedding.pkl")
    imp.put(E / "cached_dictionary_embedding.pkl",
            paths.EMBEDDING_DIR / "cached_dictionary_embedding.pkl", required=False)
    for s in SURVEYS:
        imp.put(E / f"{s}_topic_distance.csv", paths.EMBEDDING_DIR / f"{s}_topic_distance.csv")
    for s in SOCIAL:
        for kind in ("gpt", "dictionary"):
            imp.put(E / f"dynamic_embedding_{s}_{kind}.csv",
                    paths.EMBEDDING_DIR / f"dynamic_embedding_{s}_{kind}.csv")
    imp.put(E / "backup_full_question", paths.EMBEDDING_DIR / "backup_full_question")
    imp.put(E / "topic_centroids", paths.EMBEDDING_DIR / "topic_centroids", required=False)
    imp.put(E / "word_overlap_rows.json", paths.EMBEDDING_DIR / "word_overlap_rows.json", required=False)

    print("[dimension results]")
    D = L / "dimension_data"
    for f in sorted(D.glob("*.json")) + sorted(D.glob("*-loadings-summary.txt")):
        if f.name.startswith(SKIP_DIMENSION_STEMS) or f.name.startswith("twitter-none"):
            continue
        imp.put(f, paths.DIMENSION_RESULTS_DIR / f.name)

    print("[social-media csr matrices]")
    C = (args.csr_root or L / "dimension" / "data").resolve()
    for mode in TWITTER_MODES:
        imp.put(C / "twitter" / mode, paths.CSR_DIR / f"twitter-{mode}", required=False)
    imp.put(C / "weibo", paths.CSR_DIR / "weibo", required=False)

    print(f"\n{imp.copied} item(s) {'would be ' if args.dry_run else ''}"
          f"copied into {paths.display(paths.DATA_ROOT)}")
    if imp.missing:
        print(f"missing in {L} ({len(imp.missing)}):")
        for m in imp.missing:
            print(f"  {m}")


if __name__ == "__main__":
    main()
