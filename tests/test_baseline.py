import pandas as pd
from collections import defaultdict

from src.blocking.candidate_generator import (
    first_name_token,
    address_tokens,
)
from src.model.baseline import calculate_match_score


SAMPLE_SIZE = 1000
CHUNK_SIZE = 100_000


def load_sample():
    s1 = pd.read_csv(
        "data/train/train_source1.tsv",
        sep="\t",
        nrows=SAMPLE_SIZE,
    )

    gt = pd.read_csv(
        "data/train/train_ground_truth.tsv",
        sep="\t",
        nrows=SAMPLE_SIZE,
    )

    truth = {}

    for row in gt.itertuples(index=False):
        if pd.isna(row.matched_entity_ids) or not row.matched_entity_ids:
            truth[row.source1_entity_id] = set()
        else:
            truth[row.source1_entity_id] = set(
                row.matched_entity_ids.split(",")
            )

    return s1, truth


def build_s1_blocks(s1):
    name_blocks = defaultdict(set)
    address_blocks = defaultdict(set)

    for row in s1.itertuples(index=False):
        country = str(row.country).lower().strip()

        name = first_name_token(row.business_name)
        if name:
            name_blocks[(country, name)].add(row.entity_id)

        for token in address_tokens(row.business_address):
            address_blocks[(country, token)].add(row.entity_id)

    return name_blocks, address_blocks


def main():

    print("Loading 1,000 S1 records...")
    s1, truth = load_sample()

    name_blocks, address_blocks = build_s1_blocks(s1)

    s1_lookup = s1.set_index("entity_id")

    # candidate pairs stored here
    candidates = defaultdict(set)

    # Only candidate records are retained.
    candidate_records = {}

    for source_file in [
        "data/train/train_source2.tsv",
        "data/train/train_source3.tsv",
    ]:

        print("Scanning:", source_file)

        for chunk in pd.read_csv(
            source_file,
            sep="\t",
            chunksize=CHUNK_SIZE,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ],
        ):

            for row in chunk.itertuples(index=False):

                country = str(row.country).lower().strip()

                matched_s1 = set()

                # Name block
                name = first_name_token(row.business_name)

                if name:
                    matched_s1.update(
                        name_blocks.get(
                            (country, name),
                            set(),
                        )
                    )

                # Address block
                for token in address_tokens(row.business_address):
                    matched_s1.update(
                        address_blocks.get(
                            (country, token),
                            set(),
                        )
                    )

                # Save only relevant records
                for s1_id in matched_s1:
                    candidates[s1_id].add(row.entity_id)

                if matched_s1:
                    candidate_records[row.entity_id] = row

    print("\nCandidate generation complete.")

    total_candidates = sum(
        len(v) for v in candidates.values()
    )

    print("Candidate pairs:", total_candidates)
    print(
        "Average candidates/S1:",
        round(total_candidates / SAMPLE_SIZE, 2),
    )

    # ---------------------------------------------------------
    # Score candidates once.
    # ---------------------------------------------------------

    scored = []

    print("Scoring candidates...")

    for s1_id, candidate_ids in candidates.items():

        s1_row = s1_lookup.loc[s1_id]

        for candidate_id in candidate_ids:

            candidate = candidate_records[candidate_id]

            scores = calculate_match_score(
                s1_row["business_name"],
                candidate.business_name,
                s1_row["business_address"],
                candidate.business_address,
            )

            scored.append(
                (
                    s1_id,
                    candidate_id,
                    scores["final_score"],
                )
            )

    print("Scored pairs:", len(scored))

    # ---------------------------------------------------------
    # Test thresholds
    # ---------------------------------------------------------

    for threshold in [0.80, 0.85, 0.90, 0.95]:

        tp = 0
        fp = 0
        fn = 0

        for s1_id, candidate_id, score in scored:

            predicted = score >= threshold
            actual = candidate_id in truth.get(
                s1_id,
                set(),
            )

            if predicted and actual:
                tp += 1
            elif predicted and not actual:
                fp += 1
            elif not predicted and actual:
                fn += 1

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0
        )

        f05 = (
            1.25 * precision * recall
            / (0.25 * precision + recall)
            if precision + recall
            else 0
        )

        print(
            f"\nThreshold: {threshold}"
        )
        print("TP:", tp)
        print("FP:", fp)
        print("FN:", fn)
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F0.5:", round(f05, 4))


if __name__ == "__main__":
    main()