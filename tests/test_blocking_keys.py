import pandas as pd

from src.blocking.candidate_generator import (
    first_name_token,
    address_tokens,
)


SAMPLE_SIZE = 5000


def main():

    print("Loading S1 sample...")
    s1 = pd.read_csv(
        "data/train/train_source1.tsv",
        sep="\t",
        nrows=SAMPLE_SIZE,
        usecols=[
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ],
    )

    print("Loading ground truth...")
    gt = pd.read_csv(
        "data/train/train_ground_truth.tsv",
        sep="\t",
        nrows=SAMPLE_SIZE,
    )

    # Map S1 ID -> true S2/S3 IDs
    truth = {}

    for row in gt.itertuples(index=False):
        if pd.isna(row.matched_entity_ids) or not row.matched_entity_ids:
            truth[row.source1_entity_id] = set()
        else:
            truth[row.source1_entity_id] = set(
                row.matched_entity_ids.split(",")
            )

    # Collect only the true candidate IDs we need.
    true_ids = set()

    for ids in truth.values():
        true_ids.update(ids)

    print("Unique true S2/S3 IDs needed:", len(true_ids))

    # We cannot directly seek rows in a TSV, so scan in chunks,
    # but stop once all required IDs are found.
    found = {}

    for source_file in [
        "data/train/train_source2.tsv",
        "data/train/train_source3.tsv",
    ]:

        print("Searching:", source_file)

        for chunk in pd.read_csv(
            source_file,
            sep="\t",
            chunksize=100_000,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ],
        ):

            matches = chunk[
                chunk["entity_id"].isin(true_ids)
            ]

            for row in matches.itertuples(index=False):
                found[row.entity_id] = row

            if len(found) == len(true_ids):
                break

    print("True records found:", len(found))

    # --------------------------------------------------
    # Build blocking keys for S1 and true matches
    # --------------------------------------------------

    total_matches = 0
    name_hits = 0
    address_hits = 0
    either_hits = 0

    for s1_row in s1.itertuples(index=False):

        s1_id = s1_row.entity_id
        matches = truth.get(s1_id, set())

        if not matches:
            continue

        s1_country = str(
            s1_row.country
        ).strip().lower()

        s1_name_key = first_name_token(
            s1_row.business_name
        )

        s1_address_keys = set(
            address_tokens(
                s1_row.business_address
            )
        )

        for match_id in matches:

            if match_id not in found:
                continue

            match = found[match_id]

            match_country = str(
                match.country
            ).strip().lower()

            # Country must agree.
            if s1_country != match_country:
                continue

            total_matches += 1

            match_name_key = first_name_token(
                match.business_name
            )

            match_address_keys = set(
                address_tokens(
                    match.business_address
                )
            )

            name_hit = (
                s1_name_key
                and s1_name_key == match_name_key
            )

            address_hit = bool(
                s1_address_keys
                & match_address_keys
            )

            if name_hit:
                name_hits += 1

            if address_hit:
                address_hits += 1

            if name_hit or address_hit:
                either_hits += 1

    print("\n========== BLOCKING KEY RESULTS ==========")

    print("True matches checked:", total_matches)

    print(
        "Name-block hits:",
        name_hits,
        f"({name_hits / total_matches:.4f})"
        if total_matches else "",
    )

    print(
        "Address-block hits:",
        address_hits,
        f"({address_hits / total_matches:.4f})"
        if total_matches else "",
    )

    print(
        "Either-block hits:",
        either_hits,
        f"({either_hits / total_matches:.4f})"
        if total_matches else "",
    )


if __name__ == "__main__":
    main()