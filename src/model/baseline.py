import pandas as pd
from difflib import SequenceMatcher

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address,
)


def similarity(a: str, b: str) -> float:
    """Return string similarity between 0 and 1."""

    if not a or not b:
        return 0.0

    return SequenceMatcher(
        None,
        a,
        b,
    ).ratio()


def token_jaccard(a: str, b: str) -> float:
    """Return token-level Jaccard similarity."""

    if not a or not b:
        return 0.0

    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(
        a_tokens & b_tokens
    ) / len(
        a_tokens | b_tokens
    )


def calculate_match_score(
    s1_name,
    s2_name,
    s1_address,
    s2_address,
):
    """
    Calculate a simple precision-oriented matching score.

    Name:
        character similarity + token overlap

    Address:
        character similarity + token overlap

    Final:
        60% name
        40% address
    """

    name1 = normalize_business_name(
        s1_name
    )

    name2 = normalize_business_name(
        s2_name
    )

    address1 = normalize_address(
        s1_address
    )

    address2 = normalize_address(
        s2_address
    )

    # -------------------------
    # Name similarity
    # -------------------------

    name_char = similarity(
        name1,
        name2,
    )

    name_token = token_jaccard(
        name1,
        name2,
    )

    name_score = max(
        name_char,
        name_token,
    )

    # -------------------------
    # Address similarity
    # -------------------------

    address_char = similarity(
        address1,
        address2,
    )

    address_token = token_jaccard(
        address1,
        address2,
    )

    address_score = max(
        address_char,
        address_token,
    )

    # -------------------------
    # Final score
    # -------------------------

    final_score = (
        0.60 * name_score
        + 0.40 * address_score
    )

    return {
        "name_score": name_score,
        "address_score": address_score,
        "final_score": final_score,
    }


def match_candidates(
    source1_df,
    source23_df,
    candidates_df,
    threshold=0.85,
):
    """
    Score candidate pairs and return predicted matches.

    A relatively high threshold is used for V1 because
    the challenge metric is precision-heavy.
    """

    s1_lookup = source1_df.set_index(
        "entity_id"
    )

    source23_lookup = source23_df.set_index(
        "entity_id"
    )

    predictions = []

    for row in candidates_df.itertuples(
        index=False
    ):

        s1_id = row.source1_entity_id
        candidate_id = row.candidate_entity_id

        if (
            s1_id not in s1_lookup.index
            or candidate_id not in source23_lookup.index
        ):
            continue

        s1 = s1_lookup.loc[s1_id]
        candidate = source23_lookup.loc[
            candidate_id
        ]

        scores = calculate_match_score(
            s1["business_name"],
            candidate["business_name"],
            s1["business_address"],
            candidate["business_address"],
        )

        if scores["final_score"] >= threshold:

            predictions.append(
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": candidate_id,
                    "name_score": scores["name_score"],
                    "address_score": scores["address_score"],
                    "final_score": scores["final_score"],
                }
            )

    return pd.DataFrame(
        predictions
    )