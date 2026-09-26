import csv
import sqlite3
import pandas as pd

S1_PATH = "data/test/test_source1.tsv"
DB_PATH = "output/entity_index.sqlite"

MATCHING_PATH = "output/matching_results_v2.tsv"
CANDIDATE_PATH = "output/candidate_pairs_v2.tsv"

CHUNK_SIZE = 100_000

def main():
    conn = sqlite3.connect(DB_PATH)

    name_query = """
        SELECT entity_id
        FROM entities
        WHERE country=? AND norm_name=?
    """

    address_query = """
        SELECT entity_id
        FROM entities
        WHERE country=? AND norm_address=?
    """

    matching_file = open(MATCHING_PATH, "w", newline="", encoding="utf-8")
    candidate_file = open(CANDIDATE_PATH, "w", newline="", encoding="utf-8")

    mw = csv.writer(matching_file, delimiter="\t")
    cw = csv.writer(candidate_file, delimiter="\t")

    mw.writerow(["source1_entity_id", "matched_entity_ids"])
    cw.writerow(["source1_entity_id", "candidate_entity_ids"])

    processed = 0
    matched = 0

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
            country = str(row.country).strip().lower()

            # Reuse the same normalization logic stored in SQLite.
            # We need to reproduce normalization here.
            from src.preprocessing.normalize import (
                normalize_business_name,
                normalize_address,
            )

            name = normalize_business_name(row.business_name)
            address = normalize_address(row.business_address)

            name_ids = []
            address_ids = []

            if name:
                name_ids = [
                    x[0]
                    for x in conn.execute(
                        name_query, (country, name)
                    ).fetchall()
                ]

            if address:
                address_ids = [
                    x[0]
                    for x in conn.execute(
                        address_query, (country, address)
                    ).fetchall()
                ]

            name_set = set(name_ids)
            address_set = set(address_ids)

            # Candidate set = union
            candidates = list(dict.fromkeys(name_ids + address_ids))

            # Precision-first matching
            intersection = name_set & address_set

            if len(intersection) == 1:
                final_matches = list(intersection)

            elif len(name_set) == 1:
                final_matches = list(name_set)

            elif len(address_set) == 1:
                final_matches = list(address_set)

            else:
                final_matches = []

            cw.writerow([
                s1_id,
                ",".join(candidates)
            ])

            mw.writerow([
                s1_id,
                ",".join(final_matches)
            ])

            if final_matches:
                matched += 1

            processed += 1

            if processed % 100_000 == 0:
                print(
                    f"Processed {processed:,} | "
                    f"matched {matched:,}"
                )

    matching_file.close()
    candidate_file.close()
    conn.close()

    print()
    print("V2 COMPLETE")
    print("Processed:", processed)
    print("Matched:", matched)
    print("Empty:", processed - matched)
    print(MATCHING_PATH)
    print(CANDIDATE_PATH)


if __name__ == "__main__":
    main()
