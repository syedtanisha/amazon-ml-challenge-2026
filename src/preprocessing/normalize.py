"""
Reusable text normalization utilities for business entity resolution.

This module provides safe, deterministic cleaning for business names
and other text fields before blocking, feature generation, and matching.
"""

import re
import unicodedata
from typing import Any


def normalize_text(value: Any) -> str:
    """
    Normalize a text value for entity-resolution processing.

    Steps:
    - Handle missing values safely.
    - Convert to string.
    - Normalize Unicode characters.
    - Convert to lowercase.
    - Replace '&' with 'and'.
    - Remove punctuation and special characters.
    - Collapse repeated whitespace.

    Parameters
    ----------
    value:
        Input value. Can be a string, number, None, or pandas NaN.

    Returns
    -------
    str
        Cleaned normalized text.
    """
    if value is None:
        return ""

    # Handle pandas/numpy missing values without requiring pandas here.
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return ""

    # Unicode normalization.
    text = unicodedata.normalize("NFKD", text)

    # Remove Unicode combining marks while preserving the base character.
    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    # Standardize ampersand.
    text = text.replace("&", " and ")

    # Convert to lowercase.
    text = text.lower()

    # Replace punctuation/special characters with spaces.
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # Underscores are technically \w, but should behave like separators.
    text = text.replace("_", " ")

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_business_name(value: Any) -> str:
    """
    Normalize a business name using the common text-normalization rules.

    Legal suffixes and business-specific abbreviations are intentionally
    handled separately so that those rules remain reusable and testable.
    """
    return normalize_text(value)


def normalize_country(value: Any) -> str:
    """
    Normalize a country value.

    Country-specific matching rules should be handled separately from
    general text normalization.
    """
    return normalize_text(value)


def normalize_tokens(value: Any) -> list[str]:
    """
    Convert normalized text into whitespace-separated tokens.

    Empty or missing values return an empty list.
    """
    normalized = normalize_text(value)

    if not normalized:
        return []

    return normalized.split()