from collections import defaultdict
import numpy as np
import pandas as pd

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address,
)

GENERIC_NAME_TOKENS = {
    "llc", "inc", "corp", "corporation", "co", "company", "limited", "ltd",
    "plc", "the", "private", "pvt", "sarl", "sas", "smt", "sri", "shri",
    "india", "new", "global", "group", "services", "enterprise", "enterprises",
    "center", "centre", "store", "shop", "solutions", "trading", "traders",
    "tech", "technology", "technologies", "international", "holdings",
    "industries", "industry", "systems", "medical", "hospital", "clinic",
    "supermarket", "pharmacy", "mart", "bazar", "bazaar", "bhavan", "hotel",
}

GENERIC_ADDRESS_TOKENS = {
    "road", "rd", "street", "st", "floor", "avenue", "ave", "drive", "dr",
    "lane", "ln", "city", "near", "plot", "door", "building", "block",
    "district", "state", "west", "east", "north", "south", "area", "maharashtra",
    "delhi", "mumbai", "pradesh", "karnataka", "bangalore", "bengaluru", "nadu",
    "tamil", "bengal", "sector", "colony", "gujarat", "flat", "kolkata", "pune",
    "telangana", "nagpur", "hyderabad", "chennai", "ahmedabad", "surat",
    "jaipur", "lucknow", "kanpur", "indore", "thane", "bhopal", "visakhapatnam",
    "vadodara", "firozabad", "ludhiana", "rajkot", "agra", "siliguri", "nashik",
    "faridabad", "patiala", "meerut", "kalyan", "vasai", "varanasi", "srinagar",
    "dhanbad", "amritsar", "navi", "allahabad", "prayagraj", "ranchi", "howrah",
    "coimbatore", "jabalpur", "gwalior", "vijayawada", "jodhpur", "madurai",
    "raipur", "kota", "guwahati", "chandigarh", "solapur", "hubli", "bareilly",
    "moradabad", "mysore", "gurgaon", "gurugram", "aligarh", "jalandhar",
    "tiruchirappalli", "bhubaneswar", "salem", "warangal", "mira", "bhayandar",
    "thiruvananthapuram", "bhiwandi", "saharanpur", "guntur", "amravati",
    "noida", "jamshedpur", "bhilai", "cuttack", "firozpur", "kochi", "bhavnagar",
    "dehradun", "durgapur", "asansol", "nanded", "kolhapur", "ajmer", "gulbarga",
    "jamnagar", "ujjain", "loni", "silvassa", "jhansi", "ulhasnagar", "jammu",
    "sangli", "mangalore", "erode", "belgaum", "ambattur", "tirunelveli",
    "malegaon", "gaya", "jalgaon", "udaipur", "maheshtala", "davanagere",
    "kozhikode", "akola", "kurnool", "rajpur", "bokaro", "rajahmundry",
    "ballari", "agartala", "bhagalpur", "latur", "dhule", "korba", "bhilwara",
    "berhampur", "muzaffarpur", "ahmednagar", "mathura", "kollam", "avadi",
    "kadapa", "anantapur", "kamarhati", "bilaspur", "sambalpur", "shahjahanpur",
    "satara", "bijapur", "ramagundam", "shimoga", "chandrapur", "junagadh",
    "thrissur", "alwar", "bardhaman", "kultan", "nizamabad", "parbhani",
    "tumkur", "khammam", "uzhavar", "panipat", "darbhanga", "bally", "aizawl",
    "dewas", "ichalkaranji", "karnal", "bathinda", "jalna", "eluru", "barasat",
    "kirari", "purnia", "satna", "mau", "sonipat", "farrukhabad", "sagar",
    "rourkela", "durg", "imphal", "ratlam", "hapur", "arrah", "karimnagar",
    "anand", "etawah", "ambernath", "north", "south", "east", "west", "central",
    "main", "opposite", "opp", "behind", "beside", "near", "next", "front",
    "side", "post", "office", "po", "box", "pin", "code", "no", "number",
}


def first_name_token(name: str) -> str:
    norm = normalize_business_name(name)
    if not norm:
        return ""
    for token in norm.split():
        if len(token) >= 3 and token not in GENERIC_NAME_TOKENS:
            return token
    return ""


def name_prefix(name: str, length: int = 4) -> str:
    norm = normalize_business_name(name).replace(" ", "")
    if len(norm) >= length:
        return norm[:length]
    return ""


def address_tokens_v3(address: str) -> list[str]:
    norm = normalize_address(address)
    if not norm:
        return []
    res = []
    for token in norm.split():
        if len(token) >= 3 and token not in GENERIC_ADDRESS_TOKENS:
            res.append(token)
    return res


def extract_target_keys(source1_df: pd.DataFrame) -> tuple[set, set, set, set, set]:
    """
    Extract all 5 blocking keys required by Source 1 entities.
    """
    s1_nt = set()
    s1_at = set()
    s1_np = set()
    s1_en = set()
    s1_ea = set()

    countries = source1_df["country"].fillna("").astype(str).str.strip().str.lower().to_numpy()
    names = source1_df["business_name"].fillna("").astype(str).to_numpy()
    addresses = source1_df["business_address"].fillna("").astype(str).to_numpy()

    for i in range(len(names)):
        c = countries[i]
        raw_n = names[i]
        raw_a = addresses[i]

        norm_n = normalize_business_name(raw_n)
        norm_a = normalize_address(raw_a)

        nt = first_name_token(raw_n)
        if nt:
            s1_nt.add((c, nt))

        for at in address_tokens_v3(raw_a):
            s1_at.add((c, at))

        np_val = name_prefix(raw_n, 4)
        if np_val:
            s1_np.add((c, np_val))

        if norm_n:
            s1_en.add((c, norm_n))

        if norm_a:
            s1_ea.add((c, norm_a))

    return s1_nt, s1_at, s1_np, s1_en, s1_ea


def build_indexes_v3(
    source_df: pd.DataFrame,
    target_keys: tuple = None,
    max_bucket_size: int = 5000,
):
    """
    Build 5 blocking indexes for candidate retrieval.
    If target_keys is provided, only index records whose blocking keys match target_keys.
    """
    s1_nt, s1_at, s1_np, s1_en, s1_ea = target_keys if target_keys else (None, None, None, None, None)

    entity_ids = source_df["entity_id"].to_numpy()
    countries = source_df["country"].fillna("").astype(str).str.strip().str.lower().to_numpy()
    names = source_df["business_name"].fillna("").astype(str).to_numpy()
    addresses = source_df["business_address"].fillna("").astype(str).to_numpy()

    name_token_index = defaultdict(list)
    address_token_index = defaultdict(list)
    name_prefix_index = defaultdict(list)
    exact_name_index = defaultdict(list)
    exact_address_index = defaultdict(list)

    for i in range(len(entity_ids)):
        eid = entity_ids[i]
        c = countries[i]
        raw_n = names[i]
        raw_a = addresses[i]

        norm_n = normalize_business_name(raw_n)
        norm_a = normalize_address(raw_a)

        # 1. Informative first name token
        nt = first_name_token(raw_n)
        if nt and (s1_nt is None or (c, nt) in s1_nt):
            name_token_index[(c, nt)].append(eid)

        # 2. Address tokens
        for at in address_tokens_v3(raw_a):
            if s1_at is None or (c, at) in s1_at:
                address_token_index[(c, at)].append(eid)

        # 3. Name prefix (4 chars)
        np_val = name_prefix(raw_n, 4)
        if np_val and (s1_np is None or (c, np_val) in s1_np):
            name_prefix_index[(c, np_val)].append(eid)

        # 4. Exact norm name
        if norm_n and (s1_en is None or (c, norm_n) in s1_en):
            exact_name_index[(c, norm_n)].append(eid)

        # 5. Exact norm address
        if norm_a and (s1_ea is None or (c, norm_a) in s1_ea):
            exact_address_index[(c, norm_a)].append(eid)

    def prune_index(idx):
        return {k: v for k, v in idx.items() if len(v) <= max_bucket_size}

    return (
        prune_index(name_token_index),
        prune_index(address_token_index),
        prune_index(name_prefix_index),
        exact_name_index,
        exact_address_index,
    )


def generate_candidates_v3(
    source1_df: pd.DataFrame,
    source23_df: pd.DataFrame = None,
    indexes: tuple = None,
    max_bucket_size: int = 5000,
) -> pd.DataFrame:
    """
    Generate candidate S2/S3 matches for Source 1 entities using V3 blocking rules.
    """
    if indexes is None:
        if source23_df is None:
            raise ValueError("Either source23_df or pre-built indexes must be provided.")
        target_keys = extract_target_keys(source1_df)
        indexes = build_indexes_v3(source23_df, target_keys=target_keys, max_bucket_size=max_bucket_size)

    (
        name_token_idx,
        address_token_idx,
        name_prefix_idx,
        exact_name_idx,
        exact_address_idx,
    ) = indexes

    s1_ids = source1_df["entity_id"].to_numpy()
    countries = source1_df["country"].fillna("").astype(str).str.strip().str.lower().to_numpy()
    names = source1_df["business_name"].fillna("").astype(str).to_numpy()
    addresses = source1_df["business_address"].fillna("").astype(str).to_numpy()

    s1_list = []
    cand_list = []

    for i in range(len(s1_ids)):
        s1_id = s1_ids[i]
        c = countries[i]
        raw_n = names[i]
        raw_a = addresses[i]

        norm_n = normalize_business_name(raw_n)
        norm_a = normalize_address(raw_a)

        candidates = set()

        # Rule 1: Country + first name token
        nt = first_name_token(raw_n)
        if nt:
            candidates.update(name_token_idx.get((c, nt), []))

        # Rule 2: Country + address tokens
        for at in address_tokens_v3(raw_a):
            candidates.update(address_token_idx.get((c, at), []))

        # Rule 3: Country + name prefix
        np_val = name_prefix(raw_n, 4)
        if np_val:
            candidates.update(name_prefix_idx.get((c, np_val), []))

        # Rule 4: Exact norm name
        if norm_n:
            candidates.update(exact_name_idx.get((c, norm_n), []))

        # Rule 5: Exact norm address
        if norm_a:
            candidates.update(exact_address_idx.get((c, norm_a), []))

        for cand_id in candidates:
            s1_list.append(s1_id)
            cand_list.append(cand_id)

    return pd.DataFrame(
        {
            "source1_entity_id": s1_list,
            "candidate_entity_id": cand_list,
        }
    )
