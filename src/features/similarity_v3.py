from difflib import SequenceMatcher
import pandas as pd

from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_address,
)


def char_similarity(a: str, b: str) -> float:
    """SequenceMatcher character ratio between two normalized strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def token_jaccard(a: str, b: str) -> float:
    """Token-level Jaccard similarity."""
    if not a or not b:
        return 0.0
    s1 = set(a.split())
    s2 = set(b.split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def token_overlap(a: str, b: str) -> float:
    """Symmetric token overlap ratio: 2 * |A & B| / (|A| + |B|)."""
    if not a or not b:
        return 0.0
    s1 = set(a.split())
    s2 = set(b.split())
    denom = len(s1) + len(s2)
    if denom == 0:
        return 0.0
    return (2.0 * len(s1 & s2)) / denom


def prefix_similarity(a: str, b: str) -> float:
    """Fraction of shared initial characters relative to max string length."""
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 0.0
    common = 0
    for c1, c2 in zip(a, b):
        if c1 == c2:
            common += 1
        else:
            break
    return common / max_len


def shared_tokens_count(a: str, b: str) -> int:
    """Count of distinct shared tokens."""
    if not a or not b:
        return 0
    return len(set(a.split()) & set(b.split()))


FEATURE_NAMES = [
    # Business name features
    "name_char_sim",
    "name_seq_ratio",
    "name_token_jaccard",
    "name_token_overlap",
    "name_exact_match",
    "name_prefix_sim",
    # Address features
    "addr_char_sim",
    "addr_token_jaccard",
    "addr_token_overlap",
    "addr_exact_match",
    # Meta / Interaction features
    "same_country",
    "name_len_diff",
    "addr_len_diff",
    "name_shared_tokens",
    "addr_shared_tokens",
    "total_shared_tokens",
]


def extract_pair_features(
    s1_name: str,
    s23_name: str,
    s1_addr: str,
    s23_addr: str,
    s1_country: str,
    s23_country: str,
) -> dict[str, float]:
    """
    Compute comprehensive similarity features for a single candidate pair.
    Handles empty/NaN strings safely.
    """

    n1 = normalize_business_name(s1_name if pd.notna(s1_name) else "")
    n2 = normalize_business_name(s23_name if pd.notna(s23_name) else "")

    a1 = normalize_address(s1_addr if pd.notna(s1_addr) else "")
    a2 = normalize_address(s23_addr if pd.notna(s23_addr) else "")

    c1 = str(s1_country).strip().lower() if pd.notna(s1_country) else ""
    c2 = str(s23_country).strip().lower() if pd.notna(s23_country) else ""

    # Business Name Features
    n_seq = char_similarity(n1, n2)
    n_jaccard = token_jaccard(n1, n2)
    n_overlap = token_overlap(n1, n2)
    n_exact = 1.0 if (n1 and n1 == n2) else 0.0
    n_prefix = prefix_similarity(n1, n2)

    # Address Features
    a_seq = char_similarity(a1, a2)
    a_jaccard = token_jaccard(a1, a2)
    a_overlap = token_overlap(a1, a2)
    a_exact = 1.0 if (a1 and a1 == a2) else 0.0

    # Interaction & Structural Features
    same_c = 1.0 if (c1 and c1 == c2) else 0.0
    n_len_diff = float(abs(len(n1) - len(n2)))
    a_len_diff = float(abs(len(a1) - len(a2)))

    n_shared_tok = shared_tokens_count(n1, n2)
    a_shared_tok = shared_tokens_count(a1, a2)

    return {
        "name_char_sim": n_seq,
        "name_seq_ratio": n_seq,
        "name_token_jaccard": n_jaccard,
        "name_token_overlap": n_overlap,
        "name_exact_match": n_exact,
        "name_prefix_sim": n_prefix,
        "addr_char_sim": a_seq,
        "addr_token_jaccard": a_jaccard,
        "addr_token_overlap": a_overlap,
        "addr_exact_match": a_exact,
        "same_country": same_c,
        "name_len_diff": n_len_diff,
        "addr_len_diff": a_len_diff,
        "name_shared_tokens": float(n_shared_tok),
        "addr_shared_tokens": float(a_shared_tok),
        "total_shared_tokens": float(n_shared_tok + a_shared_tok),
    }
