"""Topics analysed in the paper: nine per society and data source.

The social-media pipelines collected and labelled more topics than the
analysis uses (e.g. Twitter obesity / homework, Weibo Japan / COVID policies);
statistics reported for the paper are restricted to the topics listed here.
"""

from __future__ import annotations

from typing import Dict, List, Union

US_SURVEY_TOPICS: List[str] = [
    "Abortion", "Gun", "Climate", "LGBT", "VACC",
    "DeathPenalty", "Media", "MinimumWage", "UBI",
]
CN_SURVEY_TOPICS: List[str] = [
    "Corrup", "GenderEqual", "Marriage", "Childbearing", "LGBT",
    "Environment", "Econ", "Work", "Foreign",
]
GB_SURVEY_TOPICS: List[str] = [
    "LGBT", "Abortion", "DeathPenalty", "SocialMedia", "Egalitarian",
    "Environment", "Prostitution", "HealthCare", "UnemplyAid",
]
TWITTER_TOPICS: List[str] = [
    "abo", "gun", "clc", "sxo", "vac", "dpp", "soc", "minwage", "ubi",
]
# Great Britain: Twitter/X users located in GB substitute "swe" (sexual work
# legalization) for "gun" because the EVS survey measures Prostitution, not
# gun control.
GBTWITTER_TOPICS: List[str] = [
    "abo", "swe", "clc", "sxo", "vac", "dpp", "soc", "minwage", "ubi",
]
WEIBO_TOPICS: List[int] = [7, 9, 11, 12, 10, 13, 15, 14, 0]

# ROBUSTNESS ONLY (robustness/wvs_us.py; task robust_12_wvs_us): the U.S.
# survey source measured with WVS wave 7 (USA, 2017) on items mirroring EVS.
# Never used by the main analysis. Names follow EVS where the WVS item is the
# same item (Abortion ... UnemplyAid); "WHO" (confidence in the WHO, for
# vaccine mandates) and "Press" (confidence in the press, for social media)
# are substitutes; "Environment" is the WVS environment-vs-growth item used
# for China. The robustness check uses it on eight topics (WVS_US_TOPICS
# without Prostitution, see TOPICS_8).
WVS_US_TOPICS: List[str] = [
    "Abortion", "LGBT", "DeathPenalty", "Prostitution", "Egalitarian",
    "UnemplyAid", "WHO", "Press", "Environment",
]

# ROBUSTNESS ONLY (tasks robust_12_wvs_us, robust_13_topics8): every source on
# eight topics, so that the U.S. survey can be the WVS wave 7 U.S. file
# (robust_12) or ANES without gun control (robust_13, the main sources on the
# same eight topics: separates the instrument from the topic set). Dropped: U.S. gun
# control (no WVS item) and its survey slot; British prostitution / sex-work
# legalization (no U.S. counterpart); Chinese attitudes toward the U.S. and
# its survey item (job priority for natives over immigrants), the topic least
# comparable with the other societies. Each robustness source reads its
# parent's correlation and similarity files (plotting.figures.ROBUSTNESS_PARENT).
TOPICS_8: Dict[str, List[Union[str, int]]] = {
    "wvs_us8": [t for t in WVS_US_TOPICS if t != "Prostitution"],
    "anes_8": [t for t in US_SURVEY_TOPICS if t != "Gun"],
    "wvs_8": [t for t in CN_SURVEY_TOPICS if t != "Foreign"],
    "en_evs_8": [t for t in GB_SURVEY_TOPICS if t != "Prostitution"],
    "twitter_8": [t for t in TWITTER_TOPICS if t != "gun"],
    "weibo_8": [t for t in WEIBO_TOPICS if t != 0],
    "entwitter_8": [t for t in GBTWITTER_TOPICS if t != "swe"],
}

ANALYSED_SOCIAL_TOPICS: Dict[str, List[str]] = {
    "twitter": list(dict.fromkeys(TWITTER_TOPICS + GBTWITTER_TOPICS)),   # 10 codes
    "weibo": [str(t) for t in WEIBO_TOPICS],
}


def topic_key(topic: Union[str, int]) -> str:
    """Normalise a topic label from file / folder names: 'topic-dpp' -> 'dpp', 7 -> '7'."""
    s = str(topic)
    return s[len("topic-"):] if s.startswith("topic-") else s


def is_analysed(topic: Union[str, int], platform: str) -> bool:
    return topic_key(topic) in ANALYSED_SOCIAL_TOPICS[platform]
