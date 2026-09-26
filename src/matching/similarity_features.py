import re


def normalize_text(value):
    """Basic text normalization for similarity calculations."""
    if value is None:
        return ""

    text = str(value).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize(value):
    """Convert text into a set of normalized tokens."""
    text = normalize_text(value)

    if not text:
        return set()

    return set(text.split())


def jaccard_similarity(value1, value2):
    """Calculate token-level Jaccard similarity."""
    tokens1 = tokenize(value1)
    tokens2 = tokenize(value2)

    if not tokens1 and not tokens2:
        return 0.0

    if not tokens1 or not tokens2:
        return 0.0

    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def token_overlap(value1, value2):
    """Calculate overlap relative to the smaller token set."""
    tokens1 = tokenize(value1)
    tokens2 = tokenize(value2)

    if not tokens1 or not tokens2:
        return 0.0

    return len(tokens1 & tokens2) / min(len(tokens1), len(tokens2))


def edit_similarity(value1, value2):
    """Calculate normalized Levenshtein similarity."""
    text1 = normalize_text(value1)
    text2 = normalize_text(value2)

    if not text1 and not text2:
        return 0.0

    if not text1 or not text2:
        return 0.0

    previous = list(range(len(text2) + 1))

    for i, char1 in enumerate(text1, start=1):
        current = [i]

        for j, char2 in enumerate(text2, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (char1 != char2)

            current.append(min(insertion, deletion, substitution))

        previous = current

    distance = previous[-1]
    max_length = max(len(text1), len(text2))

    return 1.0 - (distance / max_length)

def calculate_pair_features(record1, record2):
    """
    Calculate similarity features for one S1-S2/S3 candidate pair.
    """

    name1 = record1.get("business_name", "")
    name2 = record2.get("business_name", "")

    address1 = record1.get("business_address", "")
    address2 = record2.get("business_address", "")

    country1 = record1.get("country", "")
    country2 = record2.get("country", "")

    normalized_name1 = normalize_text(name1)
    normalized_name2 = normalize_text(name2)

    normalized_address1 = normalize_text(address1)
    normalized_address2 = normalize_text(address2)

    return {
        "name_exact": int(
            bool(normalized_name1)
            and normalized_name1 == normalized_name2
        ),

        "name_jaccard": jaccard_similarity(name1, name2),

        "name_edit_similarity": edit_similarity(name1, name2),

        "name_token_overlap": token_overlap(name1, name2),

        "address_exact": int(
            bool(normalized_address1)
            and normalized_address1 == normalized_address2
        ),

        "address_jaccard": jaccard_similarity(address1, address2),

        "address_edit_similarity": edit_similarity(address1, address2),

        "address_token_overlap": token_overlap(address1, address2),

        "country_match": int(
            bool(normalize_text(country1))
            and normalize_text(country1) == normalize_text(country2)
        ),

        "name_missing": int(
            not normalized_name1 or not normalized_name2
        ),

        "address_missing": int(
            not normalized_address1 or not normalized_address2
        ),

        "country_missing": int(
            not normalize_text(country1)
            or not normalize_text(country2)
        ),
    }