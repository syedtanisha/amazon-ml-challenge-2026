import pandas as pd
from collections import defaultdict

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address,
)


# ============================================================
# Generic tokens that are not useful for blocking
# ============================================================

GENERIC_NAME_TOKENS = {
    "llc",
    "inc",
    "corp",
    "corporation",
    "co",
    "company",
    "limited",
    "ltd",
    "plc",
    "the",
    "private",
    "pvt",
    "sarl",
    "sas",
    "smt",
    "sri",
    "shri",
    "india",
    "new",
}

GENERIC_ADDRESS_TOKENS = {
    "road",
    "rd",
    "street",
    "st",
    "floor",
    "avenue",
    "ave",
    "drive",
    "dr",
    "lane",
    "ln",
    "city",
    "near",
    "plot",
    "door",
    "building",
    "block",
    "district",
    "state",
    "west",
    "east",
    "north",
    "south",
    "area",
    "maharashtra",
    "delhi",
    "mumbai",
    "pradesh",
"karnataka",
"bangalore",
"nadu",
"tamil",
"bengal",
"sector",
"colony",
"gujarat",
"flat",
"kolkata",
"pune",
"telangana",
}


# ============================================================
# Business-name blocking token
# ============================================================

def first_name_token(name: str) -> str:
    """
    Return the first informative token from a business name.

    Generic/legal tokens such as:
    inc, llc, private, ltd, sri, shri
    are ignored.
    """

    name = normalize_business_name(name)

    if not name:
        return ""

    tokens = name.split()

    for token in tokens:
        if (
            len(token) >= 3
            and token not in GENERIC_NAME_TOKENS
        ):
            return token

    return ""


# ============================================================
# Address blocking tokens
# ============================================================

def address_tokens(address: str) -> list[str]:
    """
    Extract informative address tokens.

    Very common address words such as road, street,
    city, floor, etc. are ignored.
    """

    address = normalize_address(address)

    if not address:
        return []

    tokens = address.split()

    result = []

    for token in tokens:

        if len(token) < 3:
            continue

        if token in GENERIC_ADDRESS_TOKENS:
            continue

        result.append(token)

    return result


# ============================================================
# Build indexes for S2/S3
# ============================================================

def build_indexes(source_df: pd.DataFrame):
    """
    Build two blocking indexes.

    Name index:
        (country, informative_name_token)
            -> entity IDs

    Address index:
        (country, informative_address_token)
            -> entity IDs
    """

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    for row in source_df.itertuples(index=False):

        entity_id = row.entity_id

        country = str(
            row.country
        ).strip().lower()

        # -------------------------
        # Name index
        # -------------------------

        name_token = first_name_token(
            row.business_name
        )

        if name_token:

            key = (
                country,
                name_token,
            )

            name_index[key].append(
                entity_id
            )

        # -------------------------
        # Address index
        # -------------------------

        for token in address_tokens(
            row.business_address
        ):

            key = (
                country,
                token,
            )

            address_index[key].append(
                entity_id
            )

    return name_index, address_index


# ============================================================
# Generate candidates
# ============================================================

def generate_candidates(
    source1_df: pd.DataFrame,
    source23_df: pd.DataFrame,
    max_candidates_per_entity: int = 100,
) -> pd.DataFrame:
    """
    Generate candidate S2/S3 matches for Source 1.

    Blocking rules:

        1. Same country + informative business-name token
        2. Same country + informative address token

    Candidates from both rules are combined.
    """

    name_index, address_index = build_indexes(
        source23_df
    )

    results = []

    for row in source1_df.itertuples(
        index=False
    ):

        s1_id = row.entity_id

        country = str(
            row.country
        ).strip().lower()

        candidates = set()

        # ====================================================
        # BLOCK 1
        # Country + business-name token
        # ====================================================

        name_token = first_name_token(
            row.business_name
        )

        if name_token:

            key = (
                country,
                name_token,
            )

            candidates.update(
                name_index.get(
                    key,
                    []
                )
            )

        # ====================================================
        # BLOCK 2
        # Country + address tokens
        # ====================================================

        for token in address_tokens(
            row.business_address
        ):

            key = (
                country,
                token,
            )

            candidates.update(
                address_index.get(
                    key,
                    []
                )
            )

        # ====================================================
        # Limit candidates
        # ====================================================

        if len(candidates) > max_candidates_per_entity:

            candidates = set(
                list(candidates)[
                    :max_candidates_per_entity
                ]
            )

        # ====================================================
        # Save candidate pairs
        # ====================================================

        for candidate_id in candidates:

            results.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                }
            )

    return pd.DataFrame(
        results,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
        ],
    )