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
# 欧洲国家（包含多语言 aliases）→ ISO 代码映射
# ---------------------------------------------------------
eu_country_to_iso = {
    "albania": "AL",
    "andorra": "AD",
    "austria": "AT",
    "belarus": "BY",
    "belgium": "BE",
    "bosnia": "BA",
    "bulgaria": "BG",
    "croatia": "HR",
    "cyprus": "CY",
    "czechia": "CZ",
    "denmark": "DK",
    "estonia": "EE",
    "finland": "FI",
    "france": "FR",
    "germany": "DE",
    "greece": "GR",
    "hungary": "HU",
    "iceland": "IS",
    "ireland": "IE",
    "italy": "IT",
    "latvia": "LV",
    "liechtenstein": "LI",
    "lithuania": "LT",
    "luxembourg": "LU",
    "malta": "MT",
    "moldova": "MD",
    "monaco": "MC",
    "montenegro": "ME",
    "netherlands": "NL",
    "north macedonia": "MK",
    "norway": "NO",
    "poland": "PL",
    "portugal": "PT",
    "romania": "RO",
    "russia": "RU",
    "san marino": "SM",
    "serbia": "RS",
    "slovakia": "SK",
    "slovenia": "SI",
    "spain": "ES",
    "sweden": "SE",
    "switzerland": "CH",
    "ukraine": "UA",
    "united kingdom": "GB",
}

# 国家别名映射
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

# 创建别名到 ISO 的映射
alias_to_iso = {}
for country, aliases in eu_country_aliases.items():
    iso = eu_country_to_iso[country]
    for alias in aliases:
        alias_to_iso[alias.lower()] = iso

# ---------------------------------------------------------
# 国旗 emoji → ISO 代码映射
# ---------------------------------------------------------
emoji_to_iso = {
    "🇦🇱": "AL",  # Albania
    "🇦🇩": "AD",  # Andorra
    "🇦🇹": "AT",  # Austria
    "🇧🇾": "BY",  # Belarus
    "🇧🇪": "BE",  # Belgium
    "🇧🇦": "BA",  # Bosnia and Herzegovina
    "🇧🇬": "BG",  # Bulgaria
    "🇭🇷": "HR",  # Croatia
    "🇨🇾": "CY",  # Cyprus
    "🇨🇿": "CZ",  # Czechia
    "🇩🇰": "DK",  # Denmark
    "🇪🇪": "EE",  # Estonia
    "🇫🇮": "FI",  # Finland
    "🇫🇷": "FR",  # France
    "🇩🇪": "DE",  # Germany
    "🇬🇷": "GR",  # Greece
    "🇭🇺": "HU",  # Hungary
    "🇮🇸": "IS",  # Iceland
    "🇮🇪": "IE",  # Ireland
    "🇮🇹": "IT",  # Italy
    "🇱🇻": "LV",  # Latvia
    "🇱🇮": "LI",  # Liechtenstein
    "🇱🇹": "LT",  # Lithuania
    "🇱🇺": "LU",  # Luxembourg
    "🇲🇹": "MT",  # Malta
    "🇲🇩": "MD",  # Moldova
    "🇲🇨": "MC",  # Monaco
    "🇲🇪": "ME",  # Montenegro
    "🇳🇱": "NL",  # Netherlands
    "🇲🇰": "MK",  # North Macedonia
    "🇳🇴": "NO",  # Norway
    "🇵🇱": "PL",  # Poland
    "🇵🇹": "PT",  # Portugal
    "🇷🇴": "RO",  # Romania
    "🇷🇺": "RU",  # Russia
    "🇸🇲": "SM",  # San Marino
    "🇷🇸": "RS",  # Serbia
    "🇸🇰": "SK",  # Slovakia
    "🇸🇮": "SI",  # Slovenia
    "🇪🇸": "ES",  # Spain
    "🇸🇪": "SE",  # Sweden
    "🇨🇭": "CH",  # Switzerland
    "🇺🇦": "UA",  # Ukraine
    "🇬🇧": "GB",  # United Kingdom
    "🇻🇦": "VA",  # Vatican City
}

# ISO 代码到国家名称的映射（用于 GeoText 结果转换）
iso_to_country_name = {
    "AL": "Albania",
    "AD": "Andorra",
    "AT": "Austria",
    "BY": "Belarus",
    "BE": "Belgium",
    "BA": "Bosnia and Herzegovina",
    "BG": "Bulgaria",
    "HR": "Croatia",
    "CY": "Cyprus",
    "CZ": "Czech Republic",
    "DK": "Denmark",
    "EE": "Estonia",
    "FI": "Finland",
    "FR": "France",
    "DE": "Germany",
    "GR": "Greece",
    "HU": "Hungary",
    "IS": "Iceland",
    "IE": "Ireland",
    "IT": "Italy",
    "LV": "Latvia",
    "LI": "Liechtenstein",
    "LT": "Lithuania",
    "LU": "Luxembourg",
    "MT": "Malta",
    "MD": "Moldova",
    "MC": "Monaco",
    "ME": "Montenegro",
    "NL": "Netherlands",
    "MK": "North Macedonia",
    "NO": "Norway",
    "PL": "Poland",
    "PT": "Portugal",
    "RO": "Romania",
    "RU": "Russia",
    "SM": "San Marino",
    "RS": "Serbia",
    "SK": "Slovakia",
    "SI": "Slovenia",
    "ES": "Spain",
    "SE": "Sweden",
    "CH": "Switzerland",
    "UA": "Ukraine",
    "GB": "United Kingdom",
    "VA": "Vatican City",
}

# 强欧洲城市 → 国家映射
city_to_iso = {
    "london": "GB",
    "paris": "FR",
    "berlin": "DE",
    "rome": "IT",
    "madrid": "ES",
    "vienna": "AT",
    "amsterdam": "NL",
    "brussels": "BE",
    "stockholm": "SE",
    "copenhagen": "DK",
    "dublin": "IE",
    "oslo": "NO",
    "helsinki": "FI",
    "zurich": "CH",
    "geneva": "CH",
    "prague": "CZ",
    "budapest": "HU",
    "lisbon": "PT",
    "athens": "GR",
    "milan": "IT",
    "barcelona": "ES",
    "munich": "DE",
    "hamburg": "DE",
    "frankfurt": "DE",
    "krakow": "PL",
    "vilnius": "LT",
    "riga": "LV",
    "tallinn": "EE",
    "valencia": "ES",
    "manchester": "GB",
    "cambridge": "GB",
    "oxford": "GB",
}

prons = ["her", "she", "he", "him", "his", "they", "them", "bi", "hole", "black", "white", "gender", "fluid"]

strange = [
    "hell", "heaven", "twitter", "tiktok", "instagram", "facebook", "fuck", "planet",
    "alien", "aliens", "earth", "emotion", "mastodon", "ig", "tweet", "idk", "stardew",
    ".com", "podcast", "mcdonalds", "kfc", "universe"
]


def normalize(loc):
    """标准化位置字符串"""
    loc = loc.lower().strip()
    loc = loc.replace(",", " ")
    loc = loc.replace(".", " ")
    loc = loc.replace("/", " ")
    loc = loc.replace("-", " ")
    loc = loc.replace("_", " ")
    loc = loc.replace("|", " ")
    loc = re.sub(r"\s+", " ", loc)
    # 所有数字都去掉
    loc = re.sub(r"\d+", "", loc)
    # prons都去掉
    for word in prons:
        if word.lower() in loc:
            loc = loc.replace(word.lower(), "")
    return loc


def identify_country(location):
    """
    识别位置所属的国家，返回 ISO 代码
    如果无法确定，返回 None
    """
    loc_norm = normalize(location)
    if len(loc_norm) == 0:
        return None
    
    # 1. 检查 emoji
    for emoji, iso in emoji_to_iso.items():
        if emoji in location:
            return iso
    
    # 2. 检查国家别名
    for alias, iso in alias_to_iso.items():
        if alias in loc_norm:
            return iso
    
    # 3. 检查城市
    splitted = loc_norm.strip().split(" ")
    for word in splitted:
        if word in city_to_iso:
            return city_to_iso[word]
    
    # 4. 检查 ISO 代码（直接出现在位置中）
    for word in splitted:
        word_upper = word.upper()
        if word_upper in eu_country_to_iso.values():
            return word_upper
    
    # 5. 使用 GeoText
    try:
        geotext_result = GeoText(location)
        country_mentions = geotext_result.country_mentions
        if country_mentions:
            # GeoText 返回的是国家名称，需要转换为 ISO
            for country_name in country_mentions:
                country_name_lower = country_name.lower()
                # 先检查别名映射
                if country_name_lower in alias_to_iso:
                    return alias_to_iso[country_name_lower]
                # 再检查标准国家名称
                for country_key, iso in eu_country_to_iso.items():
                    if country_key in country_name_lower or country_name_lower in country_key:
                        return iso
    except:
        pass
    
    return None


def analyze_eu_locations_by_country():
    """
    分析欧洲位置，按国家分类
    返回字典：{ISO代码: [位置列表], "unknown": [无法确定的位置]}
    """
    # 读取欧洲位置列表
    with open(os.path.join(base_dir, "eu_location_classified.json"), "r") as f:
        eu_location_data = json.load(f)
    
    eu_locations = eu_location_data.get("eu", [])
    
    # 按国家分类
    country_locations = {}
    unknown_locations = []
    
    print(f"开始分析 {len(eu_locations)} 个欧洲位置...")
    
    for loc in eu_locations:
        if not loc or not loc.strip():
            continue
        
        iso = identify_country(loc)
        
        if iso:
            if iso not in country_locations:
                country_locations[iso] = []
            country_locations[iso].append(loc)
        else:
            unknown_locations.append(loc)
    
    # 添加 unknown 键
    if unknown_locations:
        country_locations["unknown"] = unknown_locations
    
    # 统计信息
    print("\n=== 分析结果统计 ===")
    for iso, locations in sorted(country_locations.items(), key=lambda x: len(x[1]), reverse=True):
        country_name = iso_to_country_name.get(iso, iso)
        print(f"{iso} ({country_name}): {len(locations)} 个位置")
    
    # 保存结果
    output_file = os.path.join(base_dir, "eu_location_by_country.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(country_locations, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")

    print(f"unknown: {unknown_locations}")
    
    # return country_locations


def merge_country_results():
    """
    合并 GPT 分析结果到 eu_location_by_country.json
    类似于 eu_user_analysis.py 中的 merge_and_report
    """
    # 读取现有的按国家分类的位置
    with open(os.path.join(base_dir, "eu_location_by_country.json"), "r") as f:
        country_locations = json.load(f)
    
    # 读取 LLM 分析结果
    llm_result_path = os.path.join(base_dir, "llm_location_eu_country", "llm_result.parquet")
    if not os.path.exists(llm_result_path):
        print(f"错误: 找不到 LLM 结果文件: {llm_result_path}")
        return
    
    llm_result = pd.read_parquet(llm_result_path)
    
    print("LLM 结果统计:")
    print(llm_result["result"].value_counts())
    
    # 合并结果：将 LLM 识别的国家添加到对应国家的列表中
    for _, row in llm_result.iterrows():
        location = row["location"]
        iso_code = row["result"]  # ISO 代码或 "unknown"
        
        # 跳过无效的 ISO 代码
        if not iso_code or iso_code == "None" or pd.isna(iso_code):
            iso_code = "unknown"
        
        # 确保 ISO 代码是字符串
        iso_code = str(iso_code).strip()
        
        # 标准化 UK -> GB
        if iso_code == "UK":
            iso_code = "GB"
        
        # 如果 ISO 代码不在现有字典中，创建新键
        if iso_code not in country_locations:
            country_locations[iso_code] = []
        
        # 如果位置不在该国家的列表中，添加它
        if location not in country_locations[iso_code]:
            country_locations[iso_code].append(location)
        
        # 如果 LLM 识别出了国家（不是 unknown），从 unknown 中移除该位置
        if iso_code != "unknown" and "unknown" in country_locations:
            if location in country_locations["unknown"]:
                country_locations["unknown"].remove(location)
    
    # 清理空的 unknown 列表
    if "unknown" in country_locations and len(country_locations["unknown"]) == 0:
        del country_locations["unknown"]
    
    # 统计信息
    print("\n=== 合并后统计 ===")
    for iso, locations in sorted(country_locations.items(), key=lambda x: len(x[1]), reverse=True):
        country_name = iso_to_country_name.get(iso, iso)
        print(f"{iso} ({country_name}): {len(locations)} 个位置")
    
    # 保存更新后的结果
    output_file = os.path.join(base_dir, "eu_location_by_country.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(country_locations, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")


def get_country_user_ids():
    """
    按国家获取用户 ID
    返回字典：{ISO代码: [用户ID列表]}
    类似于 eu_user_analysis.py 中的 get_eu_user_ids，但按国家分组
    """
    # 读取按国家分类的位置
    with open(os.path.join(base_dir, "eu_location_by_country.json"), "r") as f:
        country_locations = json.load(f)
    
    # 获取所有 feature 文件
    all_feature_files = glob.glob(os.path.join(feature_dir, "user-*.parquet"))
    
    # 按国家存储用户 ID
    country_user_ids = {}
    
    print(f"开始处理 {len(all_feature_files)} 个 feature 文件...")
    
    for feature_file in all_feature_files:
        df = pd.read_parquet(feature_file)
        df = df[df["location"].notna()]
        
        # 遍历每个国家
        for iso_code, locations in country_locations.items():
            if iso_code not in country_user_ids:
                country_user_ids[iso_code] = set()
            
            # 找到该国家的位置对应的用户
            country_df = df[df["location"].isin(locations)]
            country_user_ids[iso_code].update(country_df["id"].unique())
    
    # 转换为列表并统计
    print("\n=== 按国家用户统计 ===")
    country_user_ids_list = {}
    for iso_code, user_ids in country_user_ids.items():
        user_ids_list = list(user_ids)
        country_user_ids_list[iso_code] = user_ids_list
        country_name = iso_to_country_name.get(iso_code, iso_code)
        print(f"{iso_code} ({country_name}): {len(user_ids_list)} 个用户")
    
    # 保存结果
    output_file = os.path.join(base_dir, "eu_country_user_ids.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(country_user_ids_list, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {output_file}")
    
    # return country_user_ids_list


def get_gb_user_ids():
    """
    获取英国（GB）的用户 ID
    输出格式类似于 eu_user_analysis.py 中的 get_eu_user_ids
    """
    # 读取按国家分类的位置
    with open(os.path.join(base_dir, "eu_location_by_country.json"), "r") as f:
        country_locations = json.load(f)
    
    # 获取英国的位置列表
    gb_locations = country_locations.get("GB", [])
    
    if not gb_locations:
        print("警告: 未找到英国（GB）的位置列表")
        print("请先运行 'python eu_country_user_analysis.py analyze' 和 'python eu_country_user_analysis.py merge'")
        return
    
    print(f"找到 {len(gb_locations)} 个英国位置")
    
    # 获取所有 feature 文件
    all_feature_files = glob.glob(os.path.join(feature_dir, "user-*.parquet"))
    
    gb_user_ids = set()
    for feature_file in all_feature_files:
        df = pd.read_parquet(feature_file)
        df = df[df["location"].notna()]
        gb_df = df[df["location"].isin(gb_locations)]
        gb_user_ids.update(gb_df["id"].unique())
    
    print(f"gb_user_ids: {len(gb_user_ids)}")
    
    output_file = os.path.join(base_dir, "en_user_ids.json")
    with open(output_file, "w") as f:
        json.dump(list(gb_user_ids), f)
    
    print(f"结果已保存到: {output_file}")


def count_users_by_country_in_dataset(data_dir=os.path.join(OPINION_DIR, "eu")):
    """
    统计数据集中每个国家的用户数量
    
    Args:
        data_dir: 包含 merged-*.parquet 文件的目录
    
    Returns:
        生成 CSV 文件，包含每个国家的用户数量统计
    """
    # 读取按国家分类的用户 ID
    country_user_ids_file = os.path.join(base_dir, "eu_country_user_ids.json")
    if not os.path.exists(country_user_ids_file):
        print(f"错误: 找不到文件 {country_user_ids_file}")
        print("请先运行 'python eu_country_user_analysis.py user' 生成该文件")
        return
    
    print(f"[INFO] 加载国家用户 ID 映射: {country_user_ids_file}")
    with open(country_user_ids_file, "r") as f:
        country_user_ids = json.load(f)
    
    # 将所有国家的用户 ID 转为 set（int 类型）以便快速查找
    country_user_sets = {}
    for iso_code, user_ids in country_user_ids.items():
        country_user_sets[iso_code] = set(int(uid) for uid in user_ids)
    
    print(f"[INFO] 已加载 {len(country_user_sets)} 个国家的用户 ID")
    
    # 从数据集中收集所有存在的用户 ID
    merged_files = glob.glob(os.path.join(data_dir, "merged-*.parquet"))
    if not merged_files:
        print(f"错误: 在 {data_dir} 中找不到 merged-*.parquet 文件")
        return
    
    print(f"[INFO] 找到 {len(merged_files)} 个数据文件")
    
    # 收集数据集中所有存在的用户 ID（去重）
    dataset_user_ids = set()
    for file_path in merged_files:
        print(f"[INFO] 处理文件: {os.path.basename(file_path)}")
        try:
            df = pd.read_parquet(file_path, engine="fastparquet")
            # 确保 index 是 int 类型
            df.index = pd.to_numeric(df.index, errors="coerce").astype("Int64")
            # 去除 NaN
            df = df[df.index.notna()]
            df.index = df.index.astype("int64")
            # 添加到集合中
            dataset_user_ids.update(df.index.unique())
        except Exception as e:
            print(f"[WARNING] 处理文件 {file_path} 时出错: {e}")
            continue
    
    print(f"[INFO] 数据集中共有 {len(dataset_user_ids)} 个唯一用户 ID")
    
    # 统计每个国家在数据集中实际存在的用户数量
    country_counts = []
    for iso_code, user_ids_set in country_user_sets.items():
        # 计算交集：该国家的用户 ID 与数据集中的用户 ID 的交集
        users_in_dataset = user_ids_set & dataset_user_ids
        count = len(users_in_dataset)
        
        country_name = iso_to_country_name.get(iso_code, iso_code)
        country_counts.append({
            "country_iso": iso_code,
            "country_name": country_name,
            "user_count": count
        })
        
        print(f"{iso_code} ({country_name}): {count} 个用户")
    
    # 按用户数量降序排序
    country_counts.sort(key=lambda x: x["user_count"], reverse=True)
    
    # 创建 DataFrame 并保存为 CSV
    df_result = pd.DataFrame(country_counts)
    
    # 只保留 ISO 和用户数量两列（按用户要求）
    df_output = df_result[["country_iso", "country_name", "user_count"]]
    
    output_file = os.path.join(base_dir, "eu_country_user_count_in_dataset.csv")
    df_output.to_csv(output_file, index=False)
    
    print(f"\n[INFO] 结果已保存到: {output_file}")
    print(f"[INFO] 总计: {sum(c['user_count'] for c in country_counts)} 个用户分布在 {len(country_counts)} 个国家")
    
    return df_output


# ---------------------------------------------------------
if __name__ == "__main__":
    fire.Fire({
        "analyze": analyze_eu_locations_by_country,
        "merge": merge_country_results,
        "user": get_country_user_ids,
        "gb": get_gb_user_ids,
        "count": count_users_by_country_in_dataset,
    })

