from difflib import SequenceMatcher


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def token_jaccard(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    a_tokens = set(a.split())
    b_tokens = set(b.split())

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def combined_similarity(
    name1: str,
    name2: str,
    address1: str,
    address2: str,
) -> tuple[float, float, float]:

    name_sim = similarity(name1, name2)
    address_sim = similarity(address1, address2)

    name_jaccard = token_jaccard(name1, name2)
    address_jaccard = token_jaccard(address1, address2)

    name_score = max(name_sim, name_jaccard)
    address_score = max(address_sim, address_jaccard)

    # Name gets more weight than address.
    final_score = 0.60 * name_score + 0.40 * address_score

    return name_score, address_score, final_score