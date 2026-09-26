import csv
import os
import re
import sqlite3
import unicodedata

import pandas as pd


S2_PATH = "data/test/test_source2.tsv"
S3_PATH = "data/test/test_source3.tsv"
S1_PATH = "data/test/test_source1.tsv"

DB_PATH = "output/entity_index.sqlite"

MATCHING_PATH = "output/matching_results.tsv"
CANDIDATE_PATH = "output/candidate_pairs.tsv"

CHUNK_SIZE = 100_000


LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation",
    "co", "company", "ltd", "limited", "llc",
    "plc", "pvt", "private", "sarl", "sas", "sa"
}


def normalize_text(text):
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFKC", text).lower()

    text = re.sub(
        r"\b(null|nan|none)\b",
        " ",
        text
    )

    text = "".join(
        " " if unicodedata.category(ch).startswith("P") else ch
        for ch in text
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(text):
    text = normalize_text(text)

    if not text:
        return ""

    tokens = text.split()

    while tokens:

        if tokens[-1] in LEGAL_SUFFIXES:
            tokens.pop()
            continue

        if len(tokens) >= 3 and tokens[-3:] == ["s", "a", "s"]:
            tokens = tokens[:-3]
            continue

        if len(tokens) >= 2 and tokens[-2:] in (
            ["private", "limited"],
            ["pvt", "ltd"],
        ):
            tokens = tokens[:-2]
            continue

        break

    return " ".join(tokens)


def normalize_address(text):
    return normalize_text(text)


def create_database():

    os.makedirs("output", exist_ok=True)

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE entities (
            entity_id TEXT PRIMARY KEY,
            country TEXT,
            norm_name TEXT,
            norm_address TEXT
        )
    """)

    conn.commit()

    insert_sql = """
        INSERT OR IGNORE INTO entities
        (entity_id, country, norm_name, norm_address)
        VALUES (?, ?, ?, ?)
    """

    for path in [S2_PATH, S3_PATH]:

        print(f"Indexing {path}...")

        for chunk in pd.read_csv(
            path,
            sep="\t",
            chunksize=CHUNK_SIZE,
            usecols=[
                "entity_id",
                "business_name",
                "business_address",
                "country",
            ],
        ):

            rows = []

            for row in chunk.itertuples(index=False):

                rows.append(
                    (
                        row.entity_id,
                        str(row.country).strip().lower(),
                        normalize_name(row.business_name),
                        normalize_address(row.business_address),
                    )
                )

            conn.executemany(
                insert_sql,
                rows
            )

            conn.commit()

            print(
                f"  indexed {len(rows):,} rows"
            )

    print("Creating indexes...")

    conn.execute("""
        CREATE INDEX idx_name
        ON entities(country, norm_name)
    """)

    conn.execute("""
        CREATE INDEX idx_address
        ON entities(country, norm_address)
    """)

    conn.commit()

    return conn


def generate_submission(conn):

    print("Generating submission...")

    os.makedirs("output", exist_ok=True)

    matching_file = open(
        MATCHING_PATH,
        "w",
        newline="",
        encoding="utf-8",
    )

    candidate_file = open(
        CANDIDATE_PATH,
        "w",
        newline="",
        encoding="utf-8",
    )

    matching_writer = csv.writer(
        matching_file,
        delimiter="\t",
    )

    candidate_writer = csv.writer(
        candidate_file,
        delimiter="\t",
    )

    matching_writer.writerow([
        "source1_entity_id",
        "matched_entity_ids",
    ])

    candidate_writer.writerow([
        "source1_entity_id",
        "candidate_entity_ids",
    ])

    query_name = """
        SELECT entity_id
        FROM entities
        WHERE country = ?
        AND norm_name = ?
    """

    query_address = """
        SELECT entity_id
        FROM entities
        WHERE country = ?
        AND norm_address = ?
    """

    processed = 0
    matched_count = 0

    for chunk in pd.read_csv(
        S1_PATH,
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

            s1_id = row.entity_id
            country = str(
                row.country
            ).strip().lower()

            name = normalize_name(
                row.business_name
            )

            address = normalize_address(
                row.business_address
            )

            candidates = []

            # -----------------------------------------
            # 1. Exact normalized business name
            # -----------------------------------------

            if name:

                rows = conn.execute(
                    query_name,
                    (country, name),
                ).fetchall()

                candidates = [
                    r[0] for r in rows
                ]

            # -----------------------------------------
            # 2. Exact normalized address
            #    only if name found nothing
            # -----------------------------------------

            if not candidates and address:

                rows = conn.execute(
                    query_address,
                    (country, address),
                ).fetchall()

                candidates = [
                    r[0] for r in rows
                ]

            # Remove duplicates
            candidates = list(
                dict.fromkeys(candidates)
            )

            # -----------------------------------------
            # Candidate output
            # -----------------------------------------

            candidate_writer.writerow([
                s1_id,
                ",".join(candidates),
            ])

            # -----------------------------------------
            # Final matching output
            # -----------------------------------------

            matching_writer.writerow([
                s1_id,
                ",".join(candidates),
            ])

            if candidates:
                matched_count += 1

            processed += 1

            if processed % 100_000 == 0:

                print(
                    f"Processed {processed:,} "
                    f"S1 records | "
                    f"matched {matched_count:,}"
                )

    matching_file.close()
    candidate_file.close()

    print()
    print("========== COMPLETE ==========")
    print("S1 processed:", processed)
    print("S1 with matches:", matched_count)
    print("S1 without matches:", processed - matched_count)
    print("Matching:", MATCHING_PATH)
    print("Candidates:", CANDIDATE_PATH)


def main():

    print("================================")
    print("Amazon ML Challenge - V1")
    print("Exact normalized matching")
    print("================================")
    print()

    conn = create_database()

    generate_submission(conn)

    conn.close()


if __name__ == "__main__":
    main()