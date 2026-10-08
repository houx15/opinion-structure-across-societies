"""Users located in Europe outside Great Britain: eu_user_ids minus en_user_ids.

Inputs (LOCATION_DIR): eu_user_ids.json (eu_user_analysis.py user) and
en_user_ids.json (eu_country_user_analysis.py gb). Output: eu_nen_user_ids.json.

The original file was made with a one-off command that was not saved; this
script reproduces it exactly (same 272,725 ids = 539,070 EU - 266,345 GB).

    python -m cleaning.twitter.location.make_eu_nen_user_ids
"""

import json
import os

from cleaning.settings import BASE_DIR


def main():
    with open(os.path.join(BASE_DIR, "eu_user_ids.json")) as f:
        eu = json.load(f)
    with open(os.path.join(BASE_DIR, "en_user_ids.json")) as f:
        en = set(int(x) for x in json.load(f))
    eu_nen = [x for x in eu if int(x) not in en]
    path = os.path.join(BASE_DIR, "eu_nen_user_ids.json")
    with open(path, "w") as f:
        json.dump(eu_nen, f)
    print(f"eu={len(eu)} en={len(en)} eu_nen={len(eu_nen)} -> {path}")


if __name__ == "__main__":
    main()
