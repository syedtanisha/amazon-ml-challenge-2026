"""
End-to-end memory-safe training and matching pipeline for Amazon ML Challenge 2026.

This module features:
1. File-path based 2-pass streaming index construction without retaining DataFrames in memory.
2. Chunked feature extraction and memory-bounded candidate pair processing.
3. Ground-truth candidate labeling and 80/20 train/validation split by source1_entity_id.
4. Threshold tuning via EntityResolutionModelV3 and evaluate_macro_f05.
5. Incremental streaming output export for candidate_pairs.tsv and matching_results.tsv.
"""

from collections import Counter, defaultdict
from pathlib import Path
import time
import numpy as np
import pandas as pd

from src.blocking.candidate_generator import (
    DEFAULT_MAX_ADDRESS_BUCKET,
    address_tokens,
    first_name_token,
    generate_candidates_from_indexes,
    normalize_country,
)
from src.evaluation.metrics import evaluate_macro_f05
from src.features.similarity_v3 import FEATURE_NAMES, extract_pair_features
from src.model.model_v3 import EntityResolutionModelV3
from src.preprocessing.normalize import (
    normalize_address,
    normalize_business_name,
)

# ------------------------------------------------------------
# Path Configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAIN_S1_PATH = DATA_DIR / "train" / "train_source1.tsv"
TRAIN_S2_PATH = DATA_DIR / "train" / "train_source2.tsv"
TRAIN_S3_PATH = DATA_DIR / "train" / "train_source3.tsv"
TRAIN_GT_PATH = DATA_DIR / "train" / "train_ground_truth.tsv"

TEST_S1_PATH = DATA_DIR / "test" / "test_source1.tsv"
TEST_S2_PATH = DATA_DIR / "test" / "test_source2.tsv"
TEST_S3_PATH = DATA_DIR / "test" / "test_source3.tsv"

CHUNK_SIZE = 100_000
S1_PROCESS_CHUNK_SIZE = 50_000


# ------------------------------------------------------------
# Streaming Indexer (Disk-Based Two-Pass)
# ------------------------------------------------------------

def build_indexes_from_file_paths(
    source2_path: Path,
    source3_path: Path,
    chunksize: int = CHUNK_SIZE,
    max_address_bucket: int = DEFAULT_MAX_ADDRESS_BUCKET,
):
    """
    Build blocking indexes by streaming S2 and S3 directly from disk twice.
    Guarantees that DataFrame chunks are garbage collected immediately after each iteration.
    """
    name_index = defaultdict(list)
    address_index = defaultdict(list)
    exact_name_index = defaultdict(list)
    exact_address_index = defaultdict(list)
    address_frequency = Counter()

    cols = ["entity_id", "business_name", "business_address", "country"]

    # Pass 1: Stream S2/S3 to count address token frequencies
    for path in [source2_path, source3_path]:
        for chunk in pd.read_csv(
            path,
            sep="\t",
            chunksize=chunksize,
            dtype=str,
            keep_default_na=False,
            usecols=cols,
        ):
            for row in chunk.itertuples(index=False):
                country = normalize_country(row.country)
                for token in address_tokens(row.business_address):
                    address_frequency[(country, token)] += 1

    # Pass 2: Stream S2/S3 to populate blocking indexes
    for path in [source2_path, source3_path]:
        for chunk in pd.read_csv(
            path,
            sep="\t",
            chunksize=chunksize,
            dtype=str,
            keep_default_na=False,
            usecols=cols,
        ):
            for row in chunk.itertuples(index=False):
                entity_id = row.entity_id
                country = normalize_country(row.country)

                name_token = first_name_token(row.business_name)
                if name_token:
                    name_index[(country, name_token)].append(entity_id)

                normalized_name = normalize_business_name(row.business_name)
                if normalized_name:
                    exact_name_index[(country, normalized_name)].append(entity_id)

                normalized_address = normalize_address(row.business_address)
                if normalized_address:
                    exact_address_index[(country, normalized_address)].append(entity_id)

                for token in address_tokens(row.business_address):
                    key = (country, token)
                    if address_frequency[key] <= max_address_bucket:
                        address_index[key].append(entity_id)

    return (
        name_index,
        address_index,
        exact_name_index,
        exact_address_index,
        address_frequency,
    )


# ------------------------------------------------------------
# Selective Candidate Target Record Retrieval
# ------------------------------------------------------------

def retrieve_needed_s23_records(
    source2_path: Path,
    source3_path: Path,
    needed_entity_ids: set[str],
    chunksize: int = CHUNK_SIZE,
) -> dict[str, dict[str, str]]:
    """
    Scan S2 and S3 in chunks and retrieve record fields for requested entity IDs only.
    """
    records = {}

    if not needed_entity_ids:
        return records

    cols = ["entity_id", "business_name", "business_address", "country"]

    for path in [source2_path, source3_path]:
        for chunk in pd.read_csv(
            path,
            sep="\t",
            chunksize=chunksize,
            dtype=str,
            keep_default_na=False,
            usecols=cols,
        ):
            matches = chunk[chunk["entity_id"].isin(needed_entity_ids)]
            for row in matches.itertuples(index=False):
                records[row.entity_id] = {
                    "business_name": row.business_name,
                    "business_address": row.business_address,
                    "country": row.country,
                }
            if len(records) >= len(needed_entity_ids):
                break

    return records


def load_ground_truth_map(gt_path: Path) -> dict[str, set[str]]:
    """
    Load ground truth mapping from source1_entity_id -> set of matched_entity_ids.
    """
    df = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)
    gt_map = {}

    for row in df.itertuples(index=False):
        matched_str = str(row.matched_entity_ids).strip()
        if matched_str:
            gt_map[row.source1_entity_id] = set(matched_str.split(","))
        else:
            gt_map[row.source1_entity_id] = set()

    return gt_map


def compute_pair_features(
    candidates_df: pd.DataFrame,
    s1_dict: dict[str, dict[str, str]],
    s23_dict: dict[str, dict[str, str]],
) -> np.ndarray:
    """
    Compute V3 similarity features for a candidate DataFrame.
    """
    if len(candidates_df) == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    rows = []
    s1_ids = candidates_df["source1_entity_id"].values
    cand_ids = candidates_df["candidate_entity_id"].values

    for s1_id, cand_id in zip(s1_ids, cand_ids):
        s1_rec = s1_dict.get(s1_id, {})
        s23_rec = s23_dict.get(cand_id, {})

        feat_dict = extract_pair_features(
            s1_name=s1_rec.get("business_name", ""),
            s23_name=s23_rec.get("business_name", ""),
            s1_addr=s1_rec.get("business_address", ""),
            s23_addr=s23_rec.get("business_address", ""),
            s1_country=s1_rec.get("country", ""),
            s23_country=s23_rec.get("country", ""),
        )
        rows.append([feat_dict[fname] for fname in FEATURE_NAMES])

    return np.array(rows, dtype=np.float32)


# ------------------------------------------------------------
# Main Pipeline Function
# ------------------------------------------------------------

def run_pipeline():
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("==========================================================")
    print("   AMAZON ML CHALLENGE 2026: MEMORY-SAFE TRAIN & MATCH     ")
    print("==========================================================")

    # ------------------------------------------------------------
    # 1. Training Phase
    # ------------------------------------------------------------
    print("\n--- PHASE 1: TRAINING ---")
    print(f"Loading training Source 1 from {TRAIN_S1_PATH}...")
    train_s1 = pd.read_csv(
        TRAIN_S1_PATH, sep="\t", dtype=str, keep_default_na=False
    )
    print(f"Loaded {len(train_s1):,} training Source 1 entities.")

    print(f"Loading ground truth from {TRAIN_GT_PATH}...")
    ground_truth_map = load_ground_truth_map(TRAIN_GT_PATH)

    print("\nBuilding training S2+S3 blocking indexes (streaming 2-pass)...")
    t0 = time.time()
    train_indexes = build_indexes_from_file_paths(TRAIN_S2_PATH, TRAIN_S3_PATH)
    print(f"Training indexes built in {time.time() - t0:.2f} s.")

    # Deterministic train/validation split by source1_entity_id (80/20)
    unique_train_s1_ids = train_s1["entity_id"].unique()
    rng = np.random.RandomState(42)
    shuffled_s1_ids = unique_train_s1_ids.copy()
    rng.shuffle(shuffled_s1_ids)

    split_idx = int(len(shuffled_s1_ids) * 0.8)
    val_s1_id_list = list(shuffled_s1_ids[split_idx:])
    val_s1_id_set = set(val_s1_id_list)

    val_ground_truth_map = {
        s1_id: ground_truth_map.get(s1_id, set()) for s1_id in val_s1_id_list
    }

    # Process training S1 in chunks to keep memory bounded
    print("\nProcessing training S1 candidate pairs in chunks...")
    X_tr_list, y_tr_list = [], []
    val_cand_df_list, val_X_list = [], []

    for i in range(0, len(train_s1), S1_PROCESS_CHUNK_SIZE):
        s1_chunk = train_s1.iloc[i : i + S1_PROCESS_CHUNK_SIZE]

        s1_dict = {
            row.entity_id: {
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country,
            }
            for row in s1_chunk.itertuples(index=False)
        }

        # Candidate generation for this S1 chunk
        cands = generate_candidates_from_indexes(s1_chunk, *train_indexes)
        cands = cands.drop_duplicates(
            subset=["source1_entity_id", "candidate_entity_id"]
        ).reset_index(drop=True)

        if len(cands) == 0:
            continue

        # Label candidate pairs
        labels = [
            1 if cand_id in ground_truth_map.get(s1_id, set()) else 0
            for s1_id, cand_id in zip(
                cands["source1_entity_id"], cands["candidate_entity_id"]
            )
        ]
        cands["label"] = np.array(labels, dtype=np.int32)

        # Retrieve target records needed by this chunk's candidates
        needed_target_ids = set(cands["candidate_entity_id"])
        s23_dict = retrieve_needed_s23_records(
            TRAIN_S2_PATH, TRAIN_S3_PATH, needed_target_ids
        )

        # Compute features
        X_chunk = compute_pair_features(cands, s1_dict, s23_dict)
        y_chunk = cands["label"].values

        # Split chunk into train and val candidates
        val_mask = cands["source1_entity_id"].isin(val_s1_id_set).values
        tr_mask = ~val_mask

        if tr_mask.any():
            X_tr_list.append(X_chunk[tr_mask])
            y_tr_list.append(y_chunk[tr_mask])

        if val_mask.any():
            val_cand_df_list.append(cands[val_mask].copy())
            val_X_list.append(X_chunk[val_mask])

    # Combine split matrices
    X_tr = np.vstack(X_tr_list) if X_tr_list else np.empty((0, len(FEATURE_NAMES)))
    y_tr = np.concatenate(y_tr_list) if y_tr_list else np.empty((0,), dtype=int)
    val_candidate_df = (
        pd.concat(val_cand_df_list, ignore_index=True)
        if val_cand_df_list
        else pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
    )
    val_X = np.vstack(val_X_list) if val_X_list else np.empty((0, len(FEATURE_NAMES)))

    print(f"Train candidate pairs: {len(X_tr):,}, Val candidate pairs: {len(val_X):,}")
    print("Training EntityResolutionModelV3 on train split...")
    model_val = EntityResolutionModelV3(model_type="hist_gb")
    model_val.train(X_tr, y_tr)

    print("Tuning match probability threshold on validation split...")
    tuning_results = model_val.tune_threshold(
        val_candidate_df=val_candidate_df,
        val_X=val_X,
        val_ground_truth_map=val_ground_truth_map,
        val_s1_ids=val_s1_id_list,
    )

    selected_threshold = tuning_results["best_threshold"]
    best_metrics = tuning_results["best_metrics"]

    print("\n------------------- THRESHOLD TUNING RESULTS -------------------")
    print(f"Selected Threshold:       {selected_threshold:.2f}")
    print(f"Validation Precision:     {best_metrics['precision']:.4f}")
    print(f"Validation Recall:        {best_metrics['recall']:.4f}")
    print(f"Validation Macro F0.5:    {best_metrics['macro_f05']:.4f}")
    print("----------------------------------------------------------------")

    print("\nRetraining model on combined train + val dataset...")
    X_train_full = np.vstack([X_tr, val_X])
    y_train_full = np.concatenate([y_tr, val_candidate_df["label"].values])
    final_model = EntityResolutionModelV3(model_type="hist_gb")
    final_model.train(X_train_full, y_train_full)
    final_model.selected_threshold = selected_threshold

    # Clean up training objects to free RAM before test phase
    del train_indexes, X_tr_list, y_tr_list, val_cand_df_list, val_X_list
    del X_tr, y_tr, val_candidate_df, val_X, X_train_full, y_train_full

    # ------------------------------------------------------------
    # 2. Test Phase (Incremental Streaming Output Export)
    # ------------------------------------------------------------
    print("\n--- PHASE 2: TEST CANDIDATE GENERATION & MATCHING ---")
    print(f"Loading test Source 1 from {TEST_S1_PATH}...")
    test_s1 = pd.read_csv(
        TEST_S1_PATH, sep="\t", dtype=str, keep_default_na=False
    )
    print(f"Loaded {len(test_s1):,} test Source 1 entities.")

    print("\nBuilding test S2+S3 blocking indexes (streaming 2-pass)...")
    t0 = time.time()
    test_indexes = build_indexes_from_file_paths(TEST_S2_PATH, TEST_S3_PATH)
    print(f"Test indexes built in {time.time() - t0:.2f} s.")

    cand_pairs_out_path = OUTPUT_DIR / "candidate_pairs.tsv"
    matching_out_path = OUTPUT_DIR / "matching_results.tsv"

    # Initialize output TSV headers
    with open(cand_pairs_out_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    with open(matching_out_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

    print("\nProcessing test S1 entities in streaming chunks...")
    total_test_candidates = 0
    total_test_matches = 0

    for i in range(0, len(test_s1), S1_PROCESS_CHUNK_SIZE):
        test_s1_chunk = test_s1.iloc[i : i + S1_PROCESS_CHUNK_SIZE]

        s1_dict = {
            row.entity_id: {
                "business_name": row.business_name,
                "business_address": row.business_address,
                "country": row.country,
            }
            for row in test_s1_chunk.itertuples(index=False)
        }

        # Generate candidate pairs for test S1 chunk
        cands_chunk = generate_candidates_from_indexes(test_s1_chunk, *test_indexes)
        cands_chunk = cands_chunk.drop_duplicates(
            subset=["source1_entity_id", "candidate_entity_id"]
        ).reset_index(drop=True)

        # Build candidate mapping for this S1 chunk
        cand_map = defaultdict(list)
        for row in cands_chunk.itertuples(index=False):
            cand_map[row.source1_entity_id].append(row.candidate_entity_id)

        # Write candidate_pairs.tsv for this chunk incrementally
        with open(cand_pairs_out_path, "a", encoding="utf-8") as f_cand:
            for s1_id in test_s1_chunk["entity_id"]:
                c_list = cand_map.get(s1_id, [])
                seen = set()
                dedup_cands = []
                for c in c_list:
                    if c not in seen:
                        seen.add(c)
                        dedup_cands.append(c)
                cand_str = ",".join(dedup_cands) if dedup_cands else ""
                f_cand.write(f"{s1_id}\t{cand_str}\n")

        total_test_candidates += len(cands_chunk)

        # Retrieve needed target records for features
        needed_target_ids = set(cands_chunk["candidate_entity_id"])
        s23_dict = retrieve_needed_s23_records(
            TEST_S2_PATH, TEST_S3_PATH, needed_target_ids
        )

        # Compute features and predict
        if len(cands_chunk) > 0:
            X_test_chunk = compute_pair_features(cands_chunk, s1_dict, s23_dict)
            probs_chunk = final_model.predict_proba(X_test_chunk)
            cands_chunk["prob"] = probs_chunk

            retained_chunk = cands_chunk[
                cands_chunk["prob"] >= selected_threshold
            ]

            match_map = defaultdict(list)
            for row in retained_chunk.itertuples(index=False):
                match_map[row.source1_entity_id].append(row.candidate_entity_id)
        else:
            match_map = {}

        # Write matching_results.tsv for this chunk incrementally
        with open(matching_out_path, "a", encoding="utf-8") as f_match:
            for s1_id in test_s1_chunk["entity_id"]:
                m_list = match_map.get(s1_id, [])
                seen = set()
                dedup_matches = []
                for m in m_list:
                    if m not in seen:
                        seen.add(m)
                        dedup_matches.append(m)
                match_str = ",".join(dedup_matches) if dedup_matches else ""
                f_match.write(f"{s1_id}\t{match_str}\n")
                if match_str:
                    total_test_matches += len(dedup_matches)

        print(
            f"  Processed test chunk {i // S1_PROCESS_CHUNK_SIZE + 1} "
            f"({min(i + S1_PROCESS_CHUNK_SIZE, len(test_s1)):,}/{len(test_s1):,} S1 rows)"
        )

    total_time = time.time() - start_time
    print("\n==========================================================")
    print("      PIPELINE COMPLETED SUCCESSFULLY (MEMORY-SAFE)        ")
    print(f"Total time elapsed:         {total_time:.2f} seconds")
    print(f"Candidate pairs written:    {total_test_candidates:,}")
    print(f"Matching predictions written: {total_test_matches:,}")
    print(f"Outputs saved to:           {OUTPUT_DIR}")
    print("==========================================================")


if __name__ == "__main__":
    run_pipeline()
