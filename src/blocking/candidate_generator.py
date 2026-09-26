from collections import defaultdict, Counter

import pandas as pd

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_business_name_full,
    normalize_address,
)


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

MIN_TOKEN_LENGTH = 3

# A token appearing in more than this many target records is
# considered too common to use as a standalone address block.
DEFAULT_MAX_ADDRESS_BUCKET = 5000

# Number of rare address tokens to use for each S1 entity.
DEFAULT_ADDRESS_KEYS = 2


# ------------------------------------------------------------
# Generic name tokens
# ------------------------------------------------------------

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
    "llp",
    "the",
    "private",
    "pvt",
    "sarl",
    "sas",
    "smt",
    "sri",
    "shri",
    "india",
    "global",
    "group",
    "services",
    "enterprise",
    "enterprises",
    "center",
    "centre",
    "store",
    "shop",
    "solutions",
    "trading",
    "traders",
    "technology",
    "technologies",
    "international",
    "holdings",
    "industries",
    "industry",
}


# ------------------------------------------------------------
# Normalization helpers
# ------------------------------------------------------------

def normalize_country(country):
    if country is None:
        return ""

    return str(country).strip().lower()

def first_name_token(name):
    """
    Return the first informative business-name token.
    """

    normalized = normalize_business_name_full(name)

    if not normalized:
        return ""

    for token in normalized.split():
        if (
            len(token) >= MIN_TOKEN_LENGTH
            and token not in GENERIC_NAME_TOKENS
        ):
            return token

    return ""


def first_name_token_legal(name):
    """
    Return the first informative business-name token
    after legal suffix normalization.
    """
    return first_name_token(name)


def name_prefix(name, length=4):
    """
    Return a short normalized business-name prefix.
    """

    normalized = normalize_business_name(name)

    if not normalized:
        return ""

    normalized = normalized.replace(" ", "")

    if len(normalized) < length:
        return ""

    return normalized[:length]


def address_tokens(address):
    """
    Return normalized address tokens.

    No large hard-coded city list is used here.
    Token usefulness is determined from target-data frequency.
    """

    normalized = normalize_address(address)

    if not normalized:
        return []

    tokens = []

    for token in normalized.split():

        if len(token) < MIN_TOKEN_LENGTH:
            continue

        tokens.append(token)

    return list(dict.fromkeys(tokens))


# ------------------------------------------------------------
# Index construction
# ------------------------------------------------------------

def build_indexes(
    source23_df,
    max_address_bucket=DEFAULT_MAX_ADDRESS_BUCKET,
):
    """
    Build blocking indexes for Source 2 + Source 3.

    Returns:

        name_index
        address_index
        exact_name_index
        exact_address_index
        address_frequency
    """

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    exact_name_index = defaultdict(list)
    exact_address_index = defaultdict(list)

    address_frequency = Counter()

    # --------------------------------------------------------
    # First pass:
    # count address-token frequency
    # --------------------------------------------------------

    for row in source23_df.itertuples(index=False):

        country = normalize_country(row.country)

        for token in address_tokens(
            row.business_address
        ):
            address_frequency[
                (country, token)
            ] += 1

    # --------------------------------------------------------
    # Second pass:
    # build indexes
    # --------------------------------------------------------

    for row in source23_df.itertuples(index=False):

        entity_id = row.entity_id
        country = normalize_country(row.country)

        # -------------------------
        # Name
        # -------------------------

        name_token = first_name_token(
            row.business_name
        )

        if name_token:
            name_index[
                (country, name_token)
            ].append(entity_id)

        # -------------------------
        # Exact normalized name
        # -------------------------

        normalized_name = normalize_business_name(
            row.business_name
        )

        if normalized_name:
            exact_name_index[
                (country, normalized_name)
            ].append(entity_id)

        # -------------------------
        # Exact normalized address
        # -------------------------

        normalized_address = normalize_address(
            row.business_address
        )

        if normalized_address:
            exact_address_index[
                (country, normalized_address)
            ].append(entity_id)

        # -------------------------
        # Address tokens
        # -------------------------

        for token in address_tokens(
            row.business_address
        ):

            key = (country, token)

            # Don't create enormous buckets.
            if address_frequency[key] <= max_address_bucket:
                address_index[key].append(entity_id)

    return (
        name_index,
        address_index,
        exact_name_index,
        exact_address_index,
        address_frequency,
    )


# ------------------------------------------------------------
# Candidate generation
# ------------------------------------------------------------

def generate_candidates_from_indexes(
    source1_df,
    name_index,
    address_index,
    exact_name_index,
    exact_address_index,
    address_frequency,
    max_address_bucket=DEFAULT_MAX_ADDRESS_BUCKET,
    address_keys_per_entity=DEFAULT_ADDRESS_KEYS,
):
    """
    Generate candidate S2/S3 entities for every S1 entity using pre-built indexes.
    """
    results = []

    for row in source1_df.itertuples(index=False):

        s1_id = row.entity_id
        country = normalize_country(row.country)

        candidates = set()

        # ----------------------------------------------------
        # 1. Name blocking
        # ----------------------------------------------------

        name_token = first_name_token(
            row.business_name
        )

        if name_token:

            candidates.update(
                name_index.get(
                    (country, name_token),
                    (),
                )
            )

        # ----------------------------------------------------
        # 2. Address blocking
        # ----------------------------------------------------

        address_tokens_s1 = address_tokens(
            row.business_address
        )

        # Rank tokens by frequency.
        #
        # Lower frequency = more selective.
        #
        ranked_tokens = sorted(
            address_tokens_s1,
            key=lambda token: address_frequency.get(
                (country, token),
                float("inf"),
            ),
        )

        selected_tokens = []

        for token in ranked_tokens:

            frequency = address_frequency.get(
                (country, token),
                0,
            )

            if (
                frequency > 0
                and frequency <= max_address_bucket
            ):
                selected_tokens.append(token)

            if len(selected_tokens) >= address_keys_per_entity:
                break

        for token in selected_tokens:

            candidates.update(
                address_index.get(
                    (country, token),
                    (),
                )
            )

        # ----------------------------------------------------
        # 3. Exact name
        # ----------------------------------------------------

        normalized_name = normalize_business_name(
            row.business_name
        )

        if normalized_name:

            candidates.update(
                exact_name_index.get(
                    (country, normalized_name),
                    (),
                )
            )

        # ----------------------------------------------------
        # 4. Exact address
        # ----------------------------------------------------

        normalized_address = normalize_address(
            row.business_address
        )

        if normalized_address:

            candidates.update(
                exact_address_index.get(
                    (country, normalized_address),
                    (),
                )
            )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

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


def generate_candidates(
    source1_df,
    source23_df,
    max_address_bucket=DEFAULT_MAX_ADDRESS_BUCKET,
    address_keys_per_entity=DEFAULT_ADDRESS_KEYS,
):
    """
    Generate candidate S2/S3 entities for every S1 entity.

    Blocking strategy:

    1. Country + informative business-name token
    2. Country + rare/selective address tokens
    3. Country + exact normalized business name
    4. Country + exact normalized address

    Address tokens are ranked by their frequency in S2/S3.
    The rarest useful tokens are preferred.

    Returns:

        DataFrame with:
            source1_entity_id
            candidate_entity_id
    """

    indexes = build_indexes(
        source23_df,
        max_address_bucket=max_address_bucket,
    )

    return generate_candidates_from_indexes(
        source1_df,
        *indexes,
        max_address_bucket=max_address_bucket,
        address_keys_per_entity=address_keys_per_entity,
    )


# ------------------------------------------------------------
# Chunk-friendly index construction
# ------------------------------------------------------------

def build_indexes_from_chunks(
    source23_chunks,
    max_address_bucket=DEFAULT_MAX_ADDRESS_BUCKET,
):
    """
    Build indexes from an iterable of DataFrame chunks.

    Useful for the large 10M+ row datasets.
    """

    if not isinstance(source23_chunks, (list, tuple)):
        source23_chunks = list(source23_chunks)

    name_index = defaultdict(list)
    address_index = defaultdict(list)

    exact_name_index = defaultdict(list)
    exact_address_index = defaultdict(list)

    address_frequency = Counter()

    # --------------------------------------------------------
    # First pass: count address token frequencies
    # --------------------------------------------------------

    for chunk in source23_chunks:

        for row in chunk.itertuples(index=False):

            country = normalize_country(row.country)

            for token in address_tokens(
                row.business_address
            ):

                address_frequency[
                    (country, token)
                ] += 1

    # --------------------------------------------------------
    # Second pass: build indexes
    # --------------------------------------------------------

    for chunk in source23_chunks:

        for row in chunk.itertuples(index=False):

            entity_id = row.entity_id
            country = normalize_country(row.country)

            # Name token
            name_token = first_name_token(
                row.business_name
            )

            if name_token:
                name_index[
                    (country, name_token)
                ].append(entity_id)

            # Exact normalized name
            normalized_name = normalize_business_name(
                row.business_name
            )

            if normalized_name:
                exact_name_index[
                    (country, normalized_name)
                ].append(entity_id)

            # Exact normalized address
            normalized_address = normalize_address(
                row.business_address
            )

            if normalized_address:
                exact_address_index[
                    (country, normalized_address)
                ].append(entity_id)

            # Address tokens
            for token in address_tokens(
                row.business_address
            ):

                key = (country, token)

                if address_frequency[key] <= max_address_bucket:
                    address_index[key].append(entity_id)

    return (
        name_index,
        address_index,
        exact_name_index,
        exact_address_index,
        address_frequency,
    )