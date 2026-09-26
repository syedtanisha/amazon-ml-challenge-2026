import random
import pandas as pd

from src.matching.similarity_features import calculate_pair_features


TRAIN_DIR = "data/train"
SAMPLE_SIZE = 1000
RANDOM_SEED = 42


def load_ground_truth():
    return pd.read_csv(
        f"{TRAIN_DIR}/train_ground_truth.tsv",
        sep="\t",
        on_bad_lines="skip"
    )


def load_source1():
    return pd.read_csv(
        f"{TRAIN_DIR}/train_source1.tsv",
        sep="\t",
        on_bad_lines="skip"
    )


def get_positive_ids(ground_truth, sample_size):
    positive_pairs = []

    for _, row in ground_truth.iterrows():
        s1_id = row["source1_entity_id"]

        matched_ids = str(
            row["matched_entity_ids"]
        ).split(",")

        for entity_id in matched_ids:
            entity_id = entity_id.strip()

            if entity_id:
                positive_pairs.append(
                    (s1_id, entity_id)
                )

            if len(positive_pairs) >= sample_size:
                return positive_pairs

    return positive_pairs


def get_required_ids(positive_pairs):
    ids_by_source = {
        "S2": set(),
        "S3": set()
    }

    for _, entity_id in positive_pairs:
        if entity_id.startswith("S2-"):
            ids_by_source["S2"].add(entity_id)

        elif entity_id.startswith("S3-"):
            ids_by_source["S3"].add(entity_id)

    return ids_by_source


def load_required_records(filename, required_ids):
    records = {}

    if not required_ids:
        return records

    for chunk in pd.read_csv(
        f"{TRAIN_DIR}/{filename}",
        sep="\t",
        chunksize=100_000,
        on_bad_lines="skip"
    ):
        matched = chunk[
            chunk["entity_id"].isin(required_ids)
        ]

        for _, row in matched.iterrows():
            records[row["entity_id"]] = row.to_dict()

        if len(records) >= len(required_ids):
            break

    return records


def create_negative_pairs(
    source1_records,
    candidate_records,
    ground_truth,
    count
):
    random.seed(RANDOM_SEED)

    true_matches = {}

    for _, row in ground_truth.iterrows():
        true_matches[row["source1_entity_id"]] = set(
            str(row["matched_entity_ids"]).split(",")
        )

    candidate_ids = list(candidate_records.keys())

    negatives = []

    s1_ids = list(source1_records.keys())
    random.shuffle(s1_ids)

    for s1_id in s1_ids:

        if len(negatives) >= count:
            break

        true_ids = true_matches.get(s1_id, set())

        available = [
            entity_id
            for entity_id in candidate_ids
            if entity_id not in true_ids
        ]

        if not available:
            continue

        negative_id = random.choice(available)

        negatives.append(
            (
                s1_id,
                negative_id
            )
        )

    return negatives


def main():

    print("Loading ground truth...")
    ground_truth = load_ground_truth()

    print("Loading Source 1...")
    source1 = load_source1()

    source1_records = (
      source1
    .drop_duplicates(subset="entity_id", keep="first")
    .set_index("entity_id")
    .to_dict("index")
)

    print("Selecting positive pairs...")
    positive_pairs = get_positive_ids(
        ground_truth,
        SAMPLE_SIZE
    )

    print(
        "Positive pairs selected:",
        len(positive_pairs)
    )

    required_ids = get_required_ids(
        positive_pairs
    )

    print("Loading required S2 records...")
    s2_records = load_required_records(
        "train_source2.tsv",
        required_ids["S2"]
    )

    print("Loading required S3 records...")
    s3_records = load_required_records(
        "train_source3.tsv",
        required_ids["S3"]
    )

    candidate_records = {
        **s2_records,
        **s3_records
    }

    print(
        "Required candidate records loaded:",
        len(candidate_records)
    )

    positive_examples = []

    for s1_id, entity_id in positive_pairs:

        if (
            s1_id in source1_records
            and entity_id in candidate_records
        ):
            positive_examples.append(
                (
                    s1_id,
                    entity_id,
                    1
                )
            )

    print(
        "Valid positive examples:",
        len(positive_examples)
    )

    print("Creating negative examples...")

    negative_pairs = create_negative_pairs(
        source1_records,
        candidate_records,
        ground_truth,
        len(positive_examples)
    )

    negative_examples = [
        (s1_id, entity_id, 0)
        for s1_id, entity_id in negative_pairs
    ]

    print(
        "Negative examples:",
        len(negative_examples)
    )

    examples = (
        positive_examples +
        negative_examples
    )

    results = []

    for s1_id, entity_id, label in examples:

        features = calculate_pair_features(
            source1_records[s1_id],
            candidate_records[entity_id]
        )

        features["label"] = label

        results.append(features)

    features_df = pd.DataFrame(results)

    feature_columns = [
        column
        for column in features_df.columns
        if column != "label"
    ]

    print("\n===== FEATURE COMPARISON =====")

    comparison = (
        features_df
        .groupby("label")[feature_columns]
        .mean()
        .T
    )

    comparison.columns = [
        "Negative" if column == 0 else "Positive"
        for column in comparison.columns
    ]

    print(
        comparison.round(3).to_string()
    )

    print("\n===== FEATURE CORRELATION =====")

    print(
        features_df[feature_columns]
        .corr()
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()