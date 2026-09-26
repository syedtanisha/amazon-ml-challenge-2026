import pandas as pd
from collections import defaultdict

from src.blocking.candidate_generator import (
    first_name_token,
    first_name_token_legal,
)
from src.preprocessing.normalize import (
    normalize_business_name,
    normalize_business_name_full,
)


S1_SAMPLE_SIZE = 5000
CHUNK_SIZE = 100_000


def load_ground_truth(path):
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=S1_SAMPLE_SIZE,
        dtype=str,
        keep_default_na=False,
        usecols=["source1_entity_id", "matched_entity_ids"],
    )

    truth = {}

    for row in df.itertuples(index=False):
        if not row.matched_entity_ids:
            truth[row.source1_entity_id] = set()
        else:
            truth[row.source1_entity_id] = set(
                row.matched_entity_ids.split(",")
            )

    return truth


def name_prefix(name, length=4):
    normalized = normalize_business_name(name).replace(" ", "")
    return normalized[:length] if len(normalized) >= length else ""


def legal_name(name):
    return normalize_business_name_full(name)


def build_s1_indexes(s1):
    indexes = {
        "first_token": defaultdict(list),
        "first_token_legal": defaultdict(list),
        "name_prefix": defaultdict(list),
        "exact_name": defaultdict(list),
        "legal_name": defaultdict(list),
    }

    for row in s1.itertuples(index=False):
        s1_id = row.entity_id
        country = str(row.country).strip().lower()

        first_token = first_name_token(row.business_name)
        if first_token:
            indexes["first_token"][(country, first_token)].append(s1_id)

        first_token_legal = globals()["first_name_token_legal"](
            row.business_name
        )
        if first_token_legal:
            indexes["first_token_legal"][
                (country, first_token_legal)
            ].append(s1_id)

        prefix = name_prefix(row.business_name)
        if prefix:
            indexes["name_prefix"][(country, prefix)].append(s1_id)

        exact = normalize_business_name(row.business_name)
        if exact:
            indexes["exact_name"][(country, exact)].append(s1_id)

        legal = legal_name(row.business_name)
        if legal:
            indexes["legal_name"][(country, legal)].append(s1_id)

    return indexes


def get_keys(row):
    country = str(row.country).strip().lower()

    return {
        "first_token": (
            country,
            first_name_token(row.business_name),
        ),
        "first_token_legal": (
            country,
            first_name_token_legal(row.business_name),
        ),
        "name_prefix": (
            country,
            name_prefix(row.business_name),
        ),
        "exact_name": (
            country,
            normalize_business_name(row.business_name),
        ),
        "legal_name": (
            country,
            legal_name(row.business_name),
        ),
    }


def evaluate():
    print("Loading Source 1 sample...")
    s1 = pd.read_csv(
        "data/train/train_source1.tsv",
        sep="\t",
        nrows=S1_SAMPLE_SIZE,
        dtype=str,
        keep_default_na=False,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    print("Loading ground truth...")
    truth = load_ground_truth(
        "data/train/train_ground_truth.tsv"
    )

    indexes = build_s1_indexes(s1)

    strategies = [
        "first_token",
        "first_token_legal",
        "name_prefix",
        "exact_name",
        "legal_name",
    ]

    candidate_maps = {
        strategy: defaultdict(set)
        for strategy in strategies
    }

    total_true_matches = sum(len(matches) for matches in truth.values())

    print(f"S1 sample: {len(s1):,}")
    print(f"True matches: {total_true_matches:,}")
    print()

    for source_file in [
        "data/train/train_source2.tsv",
        "data/train/train_source3.tsv",
    ]:
        print(f"Processing {source_file}")

        processed = 0

        for chunk in pd.read_csv(
            source_file,
            sep="\t",
            chunksize=CHUNK_SIZE,
            dtype=str,
            keep_default_na=False,
            usecols=[
                "entity_id",
                "business_name",
                "country",
            ],
        ):
            for row in chunk.itertuples(index=False):
                entity_id = row.entity_id
                keys = get_keys(row)

                for strategy in strategies:
                    key = keys[strategy]

                    if not key[1]:
                        continue

                    for s1_id in indexes[strategy].get(key, ()):
                        candidate_maps[strategy][s1_id].add(entity_id)

            processed += len(chunk)
            print(f"  processed {processed:,} rows")

    print()
    print("=" * 70)
    print("NAME BLOCKING STRATEGY RESULTS")
    print("=" * 70)

    results = {}

    for strategy in strategies:
        candidate_map = candidate_maps[strategy]

        total_candidates = sum(
            len(candidate_map.get(s1_id, set()))
            for s1_id in truth
        )

        recovered_matches = sum(
            len(truth[s1_id] & candidate_map.get(s1_id, set()))
            for s1_id in truth
        )

        max_candidates = max(
            (
                len(candidate_map.get(s1_id, set()))
                for s1_id in truth
            ),
            default=0,
        )

        recall = (
            recovered_matches / total_true_matches
            if total_true_matches
            else 0
        )

        average_candidates = (
            total_candidates / len(s1)
            if len(s1)
            else 0
        )

        results[strategy] = {
            "true_matches": total_true_matches,
            "recovered_matches": recovered_matches,
            "recall": recall,
            "total_candidates": total_candidates,
            "average_candidates_per_s1": average_candidates,
            "maximum_candidates_per_s1": max_candidates,
        }

        print(f"\nStrategy: {strategy}")
        print(f"  True matches:              {total_true_matches:,}")
        print(f"  Recovered matches:         {recovered_matches:,}")
        print(f"  Recall:                    {recall:.4f}")
        print(f"  Total candidates:          {total_candidates:,}")
        print(f"  Average candidates/S1:     {average_candidates:.2f}")
        print(f"  Maximum candidates/S1:     {max_candidates:,}")

    return results


if __name__ == "__main__":
    evaluate()