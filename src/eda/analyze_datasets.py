"""
Amazon ML Challenge 2026 - Dataset EDA

Analyzes:
- Source 1, Source 2, Source 3
- Column structure and data types
- Missing values
- Duplicate records and IDs
- Business-name patterns
- Address patterns
- Country distribution
- Ground-truth matching patterns

Large TSV files are processed in chunks to avoid loading the
entire dataset into memory.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
OUTPUT_DIR = ROOT / "output" / "eda"

CHUNK_SIZE = 100_000

EXPECTED_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]


def clean_text(value: object) -> str:
    """Return a normalized string for lightweight pattern analysis."""
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def update_counter(counter: Counter, series: pd.Series) -> None:
    """Add non-empty values from a Series to a counter."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""]
    counter.update(values)


def analyze_source(path: Path) -> dict:
    """Analyze one large source TSV file chunk by chunk."""

    print(f"\nAnalyzing: {path.name}")

    total_rows = 0
    total_duplicates = 0

    missing_counts = Counter()
    country_counts = Counter()

    business_name_counts = Counter()
    address_counts = Counter()

    entity_id_counts = Counter()

    column_dtypes = {}

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
        on_bad_lines="warn",
    ):
        total_rows += len(chunk)

        # Capture columns and observed types.
        for column in chunk.columns:
            if column not in column_dtypes:
                column_dtypes[column] = str(chunk[column].dtype)

        # Missing/empty values.
        for column in chunk.columns:
            if column in EXPECTED_COLUMNS:
                missing_counts[column] += int(
                    chunk[column].isna().sum()
                    + (chunk[column].astype(str).str.strip() == "").sum()
                )

        # Duplicate rows within each chunk.
        total_duplicates += int(chunk.duplicated().sum())

        # Entity ID duplicate tracking.
        if "entity_id" in chunk.columns:
            ids = chunk["entity_id"].astype(str).str.strip()
            ids = ids[ids != ""]
            entity_id_counts.update(ids)

        # Country distribution.
        if "country" in chunk.columns:
            update_counter(country_counts, chunk["country"])

        # Business-name patterns.
        if "business_name" in chunk.columns:
            names = chunk["business_name"].map(clean_text)

            for name in names:
                if name:
                    business_name_counts[name] += 1

        # Address patterns.
        if "business_address" in chunk.columns:
            addresses = chunk["business_address"].map(clean_text)

            for address in addresses:
                if address:
                    address_counts[address] += 1

        print(
            f"  Processed {total_rows:,} rows...",
            end="\r",
            flush=True,
        )

    duplicate_entity_ids = sum(
        count - 1 for count in entity_id_counts.values() if count > 1
    )

    repeated_business_names = sum(
        count - 1 for count in business_name_counts.values() if count > 1
    )

    repeated_addresses = sum(
        count - 1 for count in address_counts.values() if count > 1
    )

    print(f"  Completed {total_rows:,} rows.")

    return {
        "file": str(path.relative_to(ROOT)),
        "rows": total_rows,
        "columns": list(column_dtypes.keys()),
        "data_types": column_dtypes,
        "missing_values": dict(missing_counts),
        "duplicate_rows_detected_within_chunks": total_duplicates,
        "duplicate_entity_ids": duplicate_entity_ids,
        "repeated_business_names": repeated_business_names,
        "repeated_addresses": repeated_addresses,
        "unique_entity_ids": len(entity_id_counts),
        "unique_business_names": len(business_name_counts),
        "unique_addresses": len(address_counts),
        "country_distribution": dict(country_counts.most_common()),
        "top_business_names": business_name_counts.most_common(20),
        "top_addresses": address_counts.most_common(20),
    }


def analyze_ground_truth(path: Path) -> dict:
    """Analyze train ground-truth matching structure."""

    print(f"\nAnalyzing ground truth: {path.name}")

    total_rows = 0

    column_counts = Counter()

    source2_match_counts = Counter()
    source3_match_counts = Counter()

    total_match_count_distribution = Counter()
    source2_match_count_distribution = Counter()
    source3_match_count_distribution = Counter()

    rows_with_source2 = 0
    rows_with_source3 = 0
    rows_with_both = 0
    rows_with_neither = 0

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=CHUNK_SIZE,
        encoding="utf-8",
        on_bad_lines="warn",
    ):
        total_rows += len(chunk)

        column_counts.update(chunk.columns)

        # The ground-truth file contains:
        # source1_entity_id
        # matched_entity_ids
        #
        # matched_entity_ids contains comma-separated S2-* and S3-* IDs.

        for matched_ids in chunk["matched_entity_ids"]:
            matched_ids = [
                item.strip()
                for item in str(matched_ids).split(",")
                if item.strip()
            ]

            source2_ids = [
                item
                for item in matched_ids
                if item.startswith("S2-")
            ]

            source3_ids = [
                item
                for item in matched_ids
                if item.startswith("S3-")
            ]

            source2_count = len(source2_ids)
            source3_count = len(source3_ids)
            total_match_count = source2_count + source3_count

            # Match-count distributions.
            total_match_count_distribution[total_match_count] += 1
            source2_match_count_distribution[source2_count] += 1
            source3_match_count_distribution[source3_count] += 1

            # Source-level presence.
            has_source2 = source2_count > 0
            has_source3 = source3_count > 0

            if has_source2:
                rows_with_source2 += 1

            if has_source3:
                rows_with_source3 += 1

            if has_source2 and has_source3:
                rows_with_both += 1

            if not has_source2 and not has_source3:
                rows_with_neither += 1

            # Track individual matched entity IDs.
            source2_match_counts.update(source2_ids)
            source3_match_counts.update(source3_ids)

        print(
            f"  Processed {total_rows:,} ground-truth rows...",
            end="\r",
            flush=True,
        )

    print(f"  Completed {total_rows:,} ground-truth rows.")

    return {
        "file": str(path.relative_to(ROOT)),
        "rows": total_rows,
        "columns": list(column_counts.keys()),

        "source2_match_columns": ["matched_entity_ids"],
        "source3_match_columns": ["matched_entity_ids"],

        "rows_with_source2_match": rows_with_source2,
        "rows_with_source3_match": rows_with_source3,
        "rows_with_both_sources": rows_with_both,
        "rows_with_neither_source": rows_with_neither,

        "total_match_count_distribution": dict(
            sorted(total_match_count_distribution.items())
        ),

        "source2_match_count_distribution": dict(
            sorted(source2_match_count_distribution.items())
        ),

        "source3_match_count_distribution": dict(
            sorted(source3_match_count_distribution.items())
        ),

        "top_source2_match_ids": source2_match_counts.most_common(20),
        "top_source3_match_ids": source3_match_counts.most_common(20),
    }


def build_report() -> dict:
    """Run the complete EDA analysis."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source_files = [
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",
    ]

    report = {
        "dataset": "Amazon ML Challenge 2026",
        "chunk_size": CHUNK_SIZE,
        "sources": {},
        "ground_truth": {},
    }

    for path in source_files:
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        report["sources"][path.stem] = analyze_source(path)

    ground_truth_path = TRAIN_DIR / "train_ground_truth.tsv"

    if not ground_truth_path.exists():
        raise FileNotFoundError(
            f"Ground-truth dataset not found: {ground_truth_path}"
        )

    report["ground_truth"] = analyze_ground_truth(ground_truth_path)

    report_path = OUTPUT_DIR / "eda_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\nEDA complete.")
    print(f"Report written to: {report_path}")

    return report


if __name__ == "__main__":
    build_report()