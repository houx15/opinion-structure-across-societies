"""Every repository-relative data and output location, in one place.

Layout (see data/README.md and outputs/README.md for file-level detail):

    data/
      survey/        survey microdata (.dta) and the survey codebook
      opinions/      cleaned opinions: individual_opinion_<survey>.parquet,
                     user_opinion_<social>_lgbt_env.parquet
      correlation/   network_analysis_<src>.csv  (pairwise r + shuffle tests)
      tf_idf/        tf_idf_<social>/ per-topic TF-IDF tables, survey_* keyword tables
      embedding/     cached embeddings, topic similarities, topic centroids
      dimension/
        csr/<stem>/<year>.csr.npz         respondent x topic sparse matrices
        results/<stem>-none-summary.json  PR / eRank / srank (+ -bootstrap.json)
    outputs/
      figures/  <YYYYMMDD>_<task>_results_<fig>.pdf
      stats/    <YYYYMMDD>_<task>_results_stats.txt
      reports/  <YYYYMMDD>_<analysis>/  (csv tables behind SI analyses)

``OSS_DATA_ROOT`` / ``OSS_OUTPUT_ROOT`` (or DATA_ROOT / OUTPUT_ROOT in
config/config.py) relocate the two roots, e.g. to a scratch disk.
"""

import os
from datetime import date
from pathlib import Path

from config import cfg

ROOT = Path(__file__).resolve().parent.parent


def _root(env: str, setting: str, default: Path) -> Path:
    value = os.environ.get(env) or getattr(cfg, setting, None)
    return Path(value).expanduser().resolve() if value else default


DATA_ROOT = _root("OSS_DATA_ROOT", "DATA_ROOT", ROOT / "data")
OUTPUT_ROOT = _root("OSS_OUTPUT_ROOT", "OUTPUT_ROOT", ROOT / "outputs")

SURVEY_DIR = DATA_ROOT / "survey"
OPINION_DIR = DATA_ROOT / "opinions"
CORRELATION_DIR = DATA_ROOT / "correlation"
TF_IDF_DIR = DATA_ROOT / "tf_idf"
EMBEDDING_DIR = DATA_ROOT / "embedding"
DIMENSION_DIR = DATA_ROOT / "dimension"
CSR_DIR = DIMENSION_DIR / "csr"
DIMENSION_RESULTS_DIR = DIMENSION_DIR / "results"
REFERENCE_DIR = DATA_ROOT / "reference"

FIGURE_DIR = OUTPUT_ROOT / "figures"
STATS_DIR = OUTPUT_ROOT / "stats"
REPORT_DIR = OUTPUT_ROOT / "reports"


def date_prefix() -> str:
    """Prefix for every file under outputs/, e.g. ``20261008``."""
    return f"{date.today():%Y%m%d}"


def report_dir(name: str, prefix: str = None) -> Path:
    """Dated folder for one SI analysis' tables: outputs/reports/<date>_<name>/."""
    path = REPORT_DIR / f"{prefix or date_prefix()}_{name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def display(path) -> str:
    """Path as written in reports: relative to the repository when possible."""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
