# Pipeline Contract

## 1. Input Records

Every business record uses the following fields:

- entity_id
- business_name
- business_address
- country

Source is determined by the input file / entity ID.

---

## 2. Preprocessing

Input:
- Source 1 records
- Source 2 records
- Source 3 records

Output:
- Cleaned Source 1 records
- Cleaned Source 2 records
- Cleaned Source 3 records

The output must preserve:
- entity_id
- business_name
- business_address
- country

---

## 3. Candidate Generation / Blocking

Input:
- Cleaned Source 1
- Cleaned Source 2
- Cleaned Source 3

Output:

- source1_entity_id
- candidate_entity_id

Candidates may only come from Source 2 or Source 3.

---

## 4. Feature Engineering

Input:
- Candidate pairs
- Corresponding business records

Output:
- Numerical similarity / matching features
- Candidate identifiers must be preserved

---

## 5. Model

Input:
- Candidate features

Output:
- Match score / prediction for each candidate pair

---

## 6. Inference

Input:
- Candidate pairs
- Model predictions

Output:

source1_entity_id
matched_entity_ids

Every Source 1 test entity must have exactly one output row.

---

## 7. Candidate Output

candidate_pairs.tsv:

source1_entity_id
candidate_entity_ids

Every final matched entity must appear in its candidate list.

---

## 8. Final Output

matching_results.tsv:

source1_entity_id
matched_entity_ids

Rules:
- One row per Source 1 entity
- No duplicate matched IDs
- Only Source 2 / Source 3 IDs
- Empty matched_entity_ids for singletons