import fire
import os
import glob
import json
import re
import pandas as pd
from geotext import GeoText
from cleaning.settings import *

base_dir = BASE_DIR
feature_dir = os.path.join(base_dir, "user_features")

# ---------------------------------------------------------
# 欧洲国家（包含多语言 aliases）
# ---------------------------------------------------------
eu_country_aliases = {
    "albania": ["albania", "shqipëria"],
    "andorra": ["andorra"],
    "austria": ["austria", "österreich"],
    "belarus": ["belarus", "byelorussia"],
    "belgium": ["belgium", "belgique", "belgië"],
    "bosnia": ["bosnia", "bosnia and herzegovina", "bih"],
    "bulgaria": ["bulgaria"],
    "croatia": ["croatia", "hrvatska"],
    "cyprus": ["cyprus"],
    "czechia": ["czechia", "czech republic", "česko", "cesko"],
    "denmark": ["denmark", "danmark"],
    "estonia": ["estonia", "eesti"],
    "finland": ["finland", "suomi"],
    "france": ["france", "francia"],
    "germany": ["germany", "deutschland"],
    "greece": ["greece", "hellas", "ελλάδα"],
    "hungary": ["hungary", "magyarország"],
    "iceland": ["iceland", "ísland"],
    "ireland": ["ireland", "éire"],
    "italy": ["italy", "italia"],
    "latvia": ["latvia", "latvija"],
    "liechtenstein": ["liechtenstein"],
    "lithuania": ["lithuania", "lietuvа"],
    "luxembourg": ["luxembourg", "letzebuerg"],
    "malta": ["malta"],
    "moldova": ["moldova"],
    "monaco": ["monaco"],
    "montenegro": ["montenegro"],
    "netherlands": ["netherlands", "holland", "nederland"],
    "north macedonia": ["north macedonia", "makedonija"],
    "norway": ["norway", "norge"],
    "poland": ["poland", "polska"],
    "portugal": ["portugal", "portuguesa"],
    "romania": ["romania", "românia"],
    "russia": ["russia", "россия"],
    "san marino": ["san marino"],
    "serbia": ["serbia", "srbija"],
    "slovakia": ["slovakia", "slovensko"],
    "slovenia": ["slovenia", "slovenija"],
    "spain": ["spain", "españa", "espana"],
    "sweden": ["sweden", "sverige"],
    "switzerland": ["switzerland", "schweiz", "suisse", "svizzera"],
    "ukraine": ["ukraine", "україна"],
    "united kingdom": ["united kingdom", "uk", "england", "scotland", "wales", "northern ireland", "gb", "britain"]
}

eu_country_names = []
for k in eu_country_aliases.keys():
    eu_country_names.append(k)

# ---------------------------------------------------------
# 国旗 emoji → ISO
# ---------------------------------------------------------
eu_country_abbrs = set([
    "AL","AD","AT","BY","BE","BA","BG","HR","CY","CZ","DK","EE","FI","FR","DE","GR",
    "HU","IS","IE","IT","LV","LI","LT","LU","MT","MD","MC","ME","NL","MK","NO","PL",
    "PT","RO","RU","SM","RS","SK","SI","ES","SE","CH","UA","GB","UK"
])

flag_emoji_regex = re.compile("[\U0001F1E6-\U0001F1FF]{2}")


prons = ["her", "she", "he", "him", "his", "they", "them", "bi", "hole", "black", "white", "gender", "fluid"]
# prons = []

strange = [
    "hell",
    "heaven",
    "twitter",
    "tiktok",
    "instagram",
    "facebook",
    "fuck",
    "planet",
    "alien",
    "aliens",
    "earth",
    "emotion",
    "mastodon",
    "ig",
    "tweet",
    "idk",
    "stardew",
    ".com",
    "podcast",
    "mcdonalds",
    "kfc",
    "universe"
]



eu_emojis = [
    "🇦🇱",  # Albania
    "🇦🇩",  # Andorra
    "🇦🇹",  # Austria
    "🇧🇾",  # Belarus
    "🇧🇪",  # Belgium
    "🇧🇦",  # Bosnia and Herzegovina
    "🇧🇬",  # Bulgaria
    "🇭🇷",  # Croatia
    "🇨🇾",  # Cyprus
    "🇨🇿",  # Czechia
    "🇩🇰",  # Denmark
    "🇪🇪",  # Estonia
    "🇫🇮",  # Finland
    "🇫🇷",  # France
    "🇩🇪",  # Germany
    "🇬🇷",  # Greece
    "🇭🇺",  # Hungary
    "🇮🇸",  # Iceland
    "🇮🇪",  # Ireland
    "🇮🇹",  # Italy
    "🇱🇻",  # Latvia
    "🇱🇮",  # Liechtenstein
    "🇱🇹",  # Lithuania
    "🇱🇺",  # Luxembourg
    "🇲🇹",  # Malta
    "🇲🇩",  # Moldova
    "🇲🇨",  # Monaco
    "🇲🇪",  # Montenegro
    "🇳🇱",  # Netherlands
    "🇲🇰",  # North Macedonia
    "🇳🇴",  # Norway
    "🇵🇱",  # Poland
    "🇵🇹",  # Portugal
    "🇷🇴",  # Romania
    "🇷🇺",  # Russia
    "🇸🇲",  # San Marino
    "🇷🇸",  # Serbia
    "🇸🇰",  # Slovakia
    "🇸🇮",  # Slovenia
    "🇪🇸",  # Spain
    "🇸🇪",  # Sweden
    "🇨🇭",  # Switzerland
    "🇺🇦",  # Ukraine
    "🇬🇧",  # United Kingdom
    "🇻🇦",  # Vatican City
]

# 强欧洲城市（强匹配）
# ---------------------------------------------------------
eu_strong_cities = {
    "london", "paris", "berlin", "rome", "madrid", "vienna", "amsterdam", "brussels",
    "stockholm", "copenhagen", "dublin", "oslo", "helsinki",
    "zurich", "geneva", "prague", "budapest", "lisbon", "athens",
    "milan", "barcelona", "munich", "hamburg", "frankfurt",
    "krakow", "vilnius", "riga", "tallinn", "valencia", "manchester", "cambridge", "oxford", "athens", "georgia", "dublin"
}


eu_strong_cities = {c.lower() for c in eu_strong_cities}

# 
non_europe_country_names = [
    "usa",
    "america",
    "united states",
    "afghanistan",
    "algeria",
    "angola",
    "antigua and barbuda",
    "argentina",
    "armenia",
    "australia",
    "azerbaijan",
    "bahamas",
    "bahrain",
    "bangladesh",
    "barbados",
    "belize",
    "benin",
    "bhutan",
    "bolivia",
    "botswana",
    "brazil",
    "brunei",
    "burkina faso",
    "burundi",
    "cabo verde",
    "cambodia",
    "cameroon",
    "canada",
    "central african republic",
    "chad",
    "chile",
    "china",
    "colombia",
    "comoros",
    "congo",
    "costa rica",
    "cuba",
    "democratic republic of the congo",
    "djibouti",
    "dominica",
    "dominican republic",
    "ecuador",
    "egypt",
    "el salvador",
    "equatorial guinea",
    "eritrea",
    "eswatini",
    "ethiopia",
    "fiji",
    "gabon",
    "gambia",
    "georgia",
    "ghana",
    "grenada",
    "guatemala",
    "guinea",
    "guinea-bissau",
    "guyana",
    "haiti",
    "honduras",
    "india",
    "indonesia",
    "iran",
    "iraq",
    "israel",
    "jamaica",
    "japan",
    "jordan",
    "kazakhstan",
    "kenya",
    "kiribati",
    "kuwait",
    "kyrgyzstan",
    "laos",
    "lebanon",
    "lesotho",
    "liberia",
    "libya",
    "madagascar",
    "malawi",
    "malaysia",
    "maldives",
    "mali",
    "marshall islands",
    "mauritania",
    "mauritius",
    "mexico",
    "micronesia",
    "mongolia",
    "morocco",
    "mozambique",
    "myanmar",
    "namibia",
    "nauru",
    "nepal",
    "new zealand",
    "nicaragua",
    "niger",
    "nigeria",
    "north korea",
    "oman",
    "pakistan",
    "palau",
    "palestine",
    "panama",
    "papua new guinea",
    "paraguay",
    "peru",
    "philippines",
    "qatar",
    "rwanda",
    "saint kitts and nevis",
    "saint lucia",
    "saint vincent and the grenadines",
    "samoa",
    "sao tome and principe",
    "saudi arabia",
    "senegal",
    "seychelles",
    "sierra leone",
    "singapore",
    "solomon islands",
    "somalia",
    "south africa",
    "south korea",
    "south sudan",
    "sri lanka",
    "sudan",
    "suriname",
    "syria",
    "taiwan",
    "tajikistan",
    "tanzania",
    "thailand",
    "timor-leste",
    "togo",
    "tonga",
    "trinidad and tobago",
    "tunisia",
    "turkey",
    "turkmenistan",
    "tuvalu",
    "uganda",
    "united arab emirates",
    "uruguay",
    "uzbekistan",
    "vanuatu",
    "venezuela",
    "vietnam",
    "yemen",
    "zambia",
    "zimbabwe",
    "africa",
    "asia",
    "oceania",
    "antarctica",
    "world",
]

non_europe_country_abbrs = [
    "AF","DZ","AO","AG","AR","AM","AU","AZ","BS","BH","BD","BB","BZ","BJ","BT","BO",
    "BW","BR","BN","BF","BI","CV","KH","CM","CA","CF","TD","CL","CN","CO","KM","CG",
    "CR","CU","CD","DJ","DM","DO","EC","EG","SV","GQ","ER","SZ","ET","FJ","GA","GM",
    "GE","GH","GD","GT","GN","GW","GY","HT","HN","IN","ID","IR","IQ","IL","JM","JP",
    "JO","KZ","KE","KI","KW","KG","LA","LB","LS","LR","LY","MG","MW","MY","MV","ML",
    "MH","MR","MU","MX","FM","MN","MA","MZ","MM","NA","NR","NP","NZ","NI","NE","NG",
    "KP","OM","PK","PW","PS","PA","PG","PY","PE","PH","QA","RW","KN","LC","VC","WS",
    "ST","SA","SN","SC","SL","SG","SB","SO","ZA","KR","SS","LK","SD","SR","SY","TW",
    "TJ","TZ","TH","TL","TG","TO","TT","TN","TR","TM","TV","UG","AE","UY","UZ","VU",
    "VE","VN","YE","ZM","ZW",
]

non_europe_country_emojis = [
    "🇦🇫","🇩🇿","🇦🇴","🇦🇬","🇦🇷","🇦🇲","🇦🇺","🇦🇿","🇧🇸","🇧🇭","🇧🇩","🇧🇧","🇧🇿",
    "🇧🇯","🇧🇹","🇧🇴","🇧🇼","🇧🇷","🇧🇳","🇧🇫","🇧🇮","🇨🇻","🇰🇭","🇨🇲","🇨🇦","🇨🇫",
    "🇹🇩","🇨🇱","🇨🇳","🇨🇴","🇰🇲","🇨🇬","🇨🇷","🇨🇺","🇨🇩","🇩🇯","🇩🇲","🇩🇴","🇪🇨",
    "🇪🇬","🇸🇻","🇬🇶","🇪🇷","🇸🇿","🇪🇹","🇫🇯","🇬🇦","🇬🇲","🇬🇪","🇬🇭","🇬🇩","🇬🇹",
    "🇬🇳","🇬🇼","🇬🇾","🇭🇹","🇭🇳","🇮🇳","🇮🇩","🇮🇷","🇮🇶","🇮🇱","🇯🇲","🇯🇵","🇯🇴",
    "🇰🇿","🇰🇪","🇰🇮","🇰🇼","🇰🇬","🇱🇦","🇱🇧","🇱🇸","🇱🇷","🇱🇾","🇲🇬","🇲🇼","🇲🇾",
    "🇲🇻","🇲🇱","🇲🇭","🇲🇷","🇲🇺","🇲🇽","🇫🇲","🇲🇳","🇲🇦","🇲🇿","🇲🇲","🇳🇦","🇳🇷",
    "🇳🇵","🇳🇿","🇳🇮","🇳🇪","🇳🇬","🇰🇵","🇴🇲","🇵🇰","🇵🇼","🇵🇸","🇵🇦","🇵🇬","🇵🇾",
    "🇵🇪","🇵🇭","🇶🇦","🇷🇼","🇰🇳","🇱🇨","🇻🇨","🇼🇸","🇸🇹","🇸🇦","🇸🇳","🇸🇨","🇸🇱",
    "🇸🇬","🇸🇧","🇸🇴","🇿🇦","🇰🇷","🇸🇸","🇱🇰","🇸🇩","🇸🇷","🇸🇾","🇹🇼","🇹🇯","🇹🇿",
    "🇹🇭","🇹🇱","🇹🇬","🇹🇴","🇹🇹","🇹🇳","🇹🇷","🇹🇲","🇹🇻","🇺🇬","🇦🇪","🇺🇾","🇺🇿",
    "🇻🇺","🇻🇪","🇻🇳","🇾🇪","🇿🇲","🇿🇼"
]


def normalize(loc):
    loc = loc.lower().strip()
    loc = loc.replace(",", " ")
    loc = loc.replace(".", " ")
    loc = loc.replace("/", " ")
    loc = loc.replace("-", " ")
    loc = loc.replace("_", " ")
    loc = loc.replace("|", " ")
    # loc = re.sub(r"[^\w\s\-\u0080-\uFFFF]", " ", loc)
    loc = re.sub(r"\s+", " ", loc)
    # 所有数字都去掉
    loc = re.sub(r"\d+", "", loc)
    # prons都去掉
    for word in prons:
        if word.lower() in loc:
            loc = loc.replace(word.lower(), "")
    # strange 都去掉
    # for word in strange:
    #     if word.lower() in loc:
    #         loc = loc.replace(word.lower(), "")
    return loc


# ---------------------------------------------------------
# classify location → {strong, weak, non_eu}
# ---------------------------------------------------------
def classify_location(location):
    loc_norm = normalize(location)
    if len(loc_norm) == 0:
        return "non_eu"
    
    for word in strange:
        if word.lower() in loc_norm:
            return "non_eu"
    
    for emoji in eu_emojis:
        if emoji in loc_norm:
            return "eu"
    
    for country in eu_country_names:
        if country.lower() in loc_norm:
            return "eu"
    
    for city in eu_strong_cities:
        if city.lower() in loc_norm:
            return "eu"

    
    for country in non_europe_country_names:
        if country.lower() in loc_norm:
            return "non_eu"
    
    for country in non_europe_country_emojis:
        if country in loc_norm:
            return "non_eu"

    splitted = (
        loc_norm.strip()
        .split(" ")
    )
    for word in splitted:
        if word.lower() in eu_country_abbrs:
            return "eu"
        if word.lower() in non_europe_country_abbrs:
            return "non_eu"
    

    # 如果没有任何 Unicode 字母，则认为无效
    if not any(ch.isalpha() for ch in loc_norm):
        return "non_eu"

    # 5) GeoText
    try:
        r = GeoText(location).country_mentions
        if r:
            for c in r:
                if normalize(c) in eu_country_abbrs:
                    return "eu"
                if normalize(c) in non_europe_country_abbrs:
                    return "non_eu"
    except:
        pass

    return "undecided"

# ---------------------------------------------------------
def identify_eu_locations():

    with open(os.path.join(base_dir, "non_us_user_analysis.json")) as f:
        data = json.load(f)

    all_locs = set(data["non_us"]) | set(data["not_sure"]) | set(data["undecided"])

    eu = set()
    unknown = set()
    non_eu = set()

    for loc in all_locs:
        if not loc.strip():
            continue

        cls = classify_location(loc)
        if cls == "eu":
            eu.add(loc)
        elif cls == "non_eu":
            non_eu.add(loc)
        else:
            unknown.add(loc)

    result = {
        "eu": list(eu),
        "unknown": list(unknown),
        "non_eu": list(non_eu)
    }

    with open(os.path.join(base_dir, "eu_location_classified.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"欧洲: {len(eu)}")
    print(f"未知: {len(unknown)}")
    print(f"非欧洲: {len(non_eu)}")

    # sample 100 unknown locations
    import random
    unknown_sample = random.sample(unknown, 100)
    for loc in unknown_sample:
        print(loc)

    return

def merge_and_report():
    with open(os.path.join(base_dir, "eu_location_classified.json"), "r") as f:
        eu_location_classified = json.load(f)

    llm_result = pd.read_parquet(os.path.join(base_dir, "llm_location_eu", "llm_result.parquet"))

    print(llm_result["result"].value_counts())

    eu_locations = set(eu_location_classified["eu"])
    llm_eu_results = llm_result[llm_result["result"] == 1]["location"]
    eu_locations.update(list(llm_eu_results))

    print(f"eu_locations: {len(eu_locations)}")

    non_eu_locations = set(eu_location_classified["non_eu"])
    llm_non_eu_results = llm_result[llm_result["result"] == 2]["location"]
    non_eu_locations.update(list(llm_non_eu_results))

    print(f"non_eu_locations: {len(non_eu_locations)}")

    undecided_locations = set(eu_location_classified["unknown"])
    llm_undecided_results = llm_result[llm_result["result"] == 0]["location"]
    undecided_locations.update(list(llm_undecided_results))

    print(f"undecided_locations: {len(undecided_locations)}")

    result = {
        "eu": list(eu_locations),
        "non_eu": list(non_eu_locations),
        "unknown": list(undecided_locations)
    }

    with open(os.path.join(base_dir, "eu_location_classified.json"), "w") as f:
        json.dump(result, f)


def get_eu_user_ids():
    with open(os.path.join(base_dir, "eu_location_classified.json"), "r") as f:
        eu_location_classified = json.load(f)
        eu_locations = eu_location_classified["eu"]

    all_feature_files = glob.glob(os.path.join(feature_dir, "user-*.parquet"))

    eu_user_ids = set()
    for feature_file in all_feature_files:
        df = pd.read_parquet(feature_file)
        df = df[df["location"].notna()]
        eu_df = df[df["location"].isin(eu_locations)]
        eu_user_ids.update(eu_df["id"].unique())

    print(f"eu_user_ids: {len(eu_user_ids)}")

    with open(os.path.join(base_dir, "eu_user_ids.json"), "w") as f:
        json.dump(list(eu_user_ids), f)


# ---------------------------------------------------------
if __name__ == "__main__":
    fire.Fire({
        "identify": identify_eu_locations,
        "merge": merge_and_report,
        "user": get_eu_user_ids,
    })