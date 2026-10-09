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
EU_SURVEY_TOPICS: List[str] = [
    "LGBT", "Abortion", "DeathPenalty", "SocialMedia", "Egalitarian",
    "Environment", "Prostitution", "HealthCare", "UnemplyAid",
]
TWITTER_TOPICS: List[str] = [
    "abo", "gun", "clc", "sxo", "vac", "dpp", "soc", "minwage", "ubi",
]
# All EU-twitter variants (eutwitter / entwitter / eu_nentwitter) substitute
# "swe" (sexual work legalization) for "gun" because the EVS survey measures
# Prostitution, not gun control.
EUTWITTER_TOPICS: List[str] = [
    "abo", "swe", "clc", "sxo", "vac", "dpp", "soc", "minwage", "ubi",
]
WEIBO_TOPICS: List[int] = [7, 9, 11, 12, 10, 13, 15, 14, 0]

ANALYSED_SOCIAL_TOPICS: Dict[str, List[str]] = {
    "twitter": list(dict.fromkeys(TWITTER_TOPICS + EUTWITTER_TOPICS)),   # 10 codes
    "weibo": [str(t) for t in WEIBO_TOPICS],
}


def topic_key(topic: Union[str, int]) -> str:
    """Normalise a topic label from file / folder names: 'topic-dpp' -> 'dpp', 7 -> '7'."""
    s = str(topic)
    return s[len("topic-"):] if s.startswith("topic-") else s


def is_analysed(topic: Union[str, int], platform: str) -> bool:
    return topic_key(topic) in ANALYSED_SOCIAL_TOPICS[platform]
