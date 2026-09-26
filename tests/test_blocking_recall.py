import pandas as pd
from collections import defaultdict

from src.blocking.candidate_generator import (
    first_name_token,
    address_tokens,
)
from src.preprocessing.normalize import normalize_business_name


SAMPLE_SIZE = 5000
CHUNK_SIZE = 100_000


def load_ground_truth(path):
    df = pd.read_csv(
        path,
        sep="\t",
        nrows=SAMPLE_SIZE,
    )

    truth = {}

    for row in df.itertuples(index=False):
        if pd.isna(row.matched_entity_ids) or not row.matched_entity_ids:
            truth[row.source1_entity_id] = set()
        else:
            truth[row.source1_entity_id] = set(
                row.matched_entity_ids.split(",")
            )

    return truth


def build_indexes_chunked(source1_df):
    """
    Build the blocking keys we need for the S1 sample.
    """

    name_keys = defaultdict(set)
    address_keys = defaultdict(set)

    for row in source1_df.itertuples(index=False):

        s1_id = row.entity_id
        country = str(row.country).strip().lower()

        token = first_name_token(row.business_name)

        if token:
            name_keys[(country, token)].add(s1_id)

        for token in address_tokens(row.business_address):
            address_keys[(country, token)].add(s1_id)

    return name_keys, address_keys


def main():

    print("Loading S1 sample...")

    s1 = pd.read_csv(
        "data/train/train_source1.tsv",
        sep="\t",
        nrows=SAMPLE_SIZE,
    )

    truth = load_ground_truth(
        "data/train/train_ground_truth.tsv"
    )

    name_keys, address_keys = build_indexes_chunked(s1)

    candidate_map = defaultdict(set)

    print("Scanning S2/S3 in chunks...")

    source_files = [
        "data/train/train_source2.tsv",
        "data/train/train_source3.tsv",
    ]

    for source_file in source_files:

        print(f"Processing {source_file}")

        for chunk in pd.read_csv(
            source_file,
            sep="\t",
            chunksize=CHUNK_SIZE,
        ):

            for row in chunk.itertuples(index=False):

                entity_id = row.entity_id
                country = str(row.country).strip().lower()

                # Name block
                name_token = first_name_token(
                    row.business_name
                )

                if name_token:

                    key = (country, name_token)

                    for s1_id in name_keys.get(key, ()):

                        candidate_map[s1_id].add(
                            entity_id
                        )

                # Address block
                for token in address_tokens(
                    row.business_address
                ):

                    key = (country, token)

                    for s1_id in address_keys.get(key, ()):

                        candidate_map[s1_id].add(
                            entity_id
                        )

            print(
                f"  processed {len(chunk):,} rows"
            )

    total_true = 0
    recovered_true = 0
    total_candidates = 0
    singleton_count = 0

    for s1_id, true_matches in truth.items():

        candidates = candidate_map.get(
            s1_id,
            set(),
        )

        total_candidates += len(candidates)

        if not true_matches:
            singleton_count += 1
            continue

        total_true += len(true_matches)

        recovered_true += len(
            true_matches & candidates
        )

    recall = (
        recovered_true / total_true
        if total_true
        else 0
    )

    print("\n========== BLOCKING RESULTS ==========")
    print("S1 sample:", SAMPLE_SIZE)
    print("Candidate pairs:", total_candidates)
    print("True matches:", total_true)
    print("Recovered true matches:", recovered_true)
    print("Blocking recall:", round(recall, 4))
    print("Singletons:", singleton_count)

    if total_candidates:
        print(
            "Average candidates/S1:",
            round(total_candidates / SAMPLE_SIZE, 2),
        )


if __name__ == "__main__":
    main()