import unittest

from src.matching.similarity_features import (
    normalize_text,
    jaccard_similarity,
    token_overlap,
    edit_similarity,
    calculate_pair_features,
)


class TestSimilarityFeatures(unittest.TestCase):

    def test_normalize_text(self):
        self.assertEqual(
            normalize_text("ABC, Store. Pvt. Ltd."),
            "abc store pvt ltd"
        )

    def test_identical_text(self):
        self.assertEqual(
            edit_similarity("ABC Store", "ABC Store"),
            1.0
        )

    def test_jaccard_similarity(self):
        score = jaccard_similarity(
            "ABC Store Pvt Ltd",
            "ABC Store"
        )
        self.assertEqual(score, 0.5)

    def test_token_overlap(self):
        score = token_overlap(
            "ABC Store Pvt Ltd",
            "ABC Store"
        )
        self.assertEqual(score, 1.0)

    def test_pair_features(self):
        record1 = {
            "business_name": "ABC Store Pvt Ltd",
            "business_address": "12 Main Street",
            "country": "India",
        }

        record2 = {
            "business_name": "ABC Store",
            "business_address": "12 Main St",
            "country": "India",
        }

        features = calculate_pair_features(record1, record2)

        self.assertIn("name_jaccard", features)
        self.assertIn("name_edit_similarity", features)
        self.assertIn("address_jaccard", features)
        self.assertIn("address_edit_similarity", features)
        self.assertIn("country_match", features)

        self.assertEqual(features["country_match"], 1)
        self.assertEqual(features["name_missing"], 0)
        self.assertEqual(features["address_missing"], 0)


if __name__ == "__main__":
    unittest.main()