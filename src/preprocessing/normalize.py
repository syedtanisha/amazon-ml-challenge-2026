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
# Common business/legal abbreviations.
LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "pvt.": "private",
    "private": "private",
    "ltd": "limited",
    "ltd.": "limited",
    "limited": "limited",
    "inc": "incorporated",
    "inc.": "incorporated",
    "incorporated": "incorporated",
    "corp": "corporation",
    "corp.": "corporation",
    "corporation": "corporation",
    "co": "company",
    "co.": "company",
    "company": "company",
    "llc": "llc",
    "l l c": "llc",
    "plc": "plc",
    "p l c": "plc",
    "llp": "llp",
    "l l p": "llp",
}


def normalize_legal_suffixes(value: Any) -> str:
    """
    Normalize common business/legal suffix abbreviations.

    Handles both normal abbreviations such as:
        Pvt. -> private
        Ltd. -> limited
        Inc. -> incorporated

    and dotted legal forms such as:
        L.L.C. -> llc
        P.L.C. -> plc
        L.L.P. -> llp
    """
    normalized = normalize_text(value)

    if not normalized:
        return ""

    tokens = normalized.split()
    result = []
    index = 0

    while index < len(tokens):

        # Handle dotted legal abbreviations after punctuation removal.
        if tokens[index:index + 3] == ["l", "l", "c"]:
            result.append("llc")
            index += 3
            continue

        if tokens[index:index + 3] == ["p", "l", "c"]:
            result.append("plc")
            index += 3
            continue

        if tokens[index:index + 3] == ["l", "l", "p"]:
            result.append("llp")
            index += 3
            continue

        # Handle normal one-token legal/business abbreviations.
        result.append(LEGAL_SUFFIX_MAP.get(tokens[index], tokens[index]))
        index += 1

    return " ".join(result)
    """
    Normalize common business/legal suffix abbreviations.

    The function first applies general text normalization and then
    replaces recognized legal/business suffix tokens.
    """
    normalized = normalize_text(value)

    if not normalized:
        return ""

    tokens = normalized.split()

    normalized_tokens = [
        LEGAL_SUFFIX_MAP.get(token, token)
        for token in tokens
    ]

    return " ".join(normalized_tokens)


def normalize_business_name_full(value: Any) -> str:
    """
    Fully normalize a business name.

    This combines:
    1. General text normalization.
    2. Legal suffix normalization.
    """
    return normalize_legal_suffixes(value)
# Common address abbreviations.
ADDRESS_ABBREVIATION_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "boul": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "hwy": "highway",
    "pkwy": "parkway",
    "ct": "court",
    "cir": "circle",
    "pl": "place",
    "ter": "terrace",
    "trl": "trail",
    "way": "way",
    "sq": "square",
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "bldg": "building",
    "rm": "room",
    "no": "number",
}


def normalize_address(value: Any) -> str:
    """
    Normalize a business address.

    Applies general text normalization followed by common
    address abbreviation expansion.
    """
    normalized = normalize_text(value)

    if not normalized:
        return ""

    tokens = normalized.split()

    normalized_tokens = [
        ADDRESS_ABBREVIATION_MAP.get(token, token)
        for token in tokens
    ]

    return " ".join(normalized_tokens)
def normalize_business_record(
    business_name: Any = None,
    business_address: Any = None,
    country: Any = None,
) -> dict[str, str]:
    """
    Normalize the main fields of a business record.

    Missing fields are represented by empty strings.

    Returns
    -------
    dict[str, str]
        Dictionary containing normalized business name,
        address, and country.
    """
    return {
        "business_name": normalize_business_name_full(business_name),
        "business_address": normalize_address(business_address),
        "country": normalize_country(country),
    }


def is_missing(value: Any) -> bool:
    """
    Check whether a value is missing or effectively empty.
    """
    if value is None:
        return True

    try:
        if value != value:
            return True
    except (TypeError, ValueError):
        pass

    return not str(value).strip()


def normalize_optional_text(value: Any) -> str:
    """
    Normalize an optional text field.

    Missing values return an empty string.
    """
    if is_missing(value):
        return ""

    return normalize_text(value)