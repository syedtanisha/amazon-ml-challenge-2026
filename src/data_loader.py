from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset directories
TRAIN_DIR = PROJECT_ROOT / "data" / "train"
TEST_DIR = PROJECT_ROOT / "data" / "test"


def load_training_data():
    """Load all training files."""

    source1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t"
    )

    source2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t"
    )

    source3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t"
    )

    ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t"
    )

    return source1, source2, source3, ground_truth


def load_test_data():
    """Load all test files."""

    source1 = pd.read_csv(
        TEST_DIR / "test_source1.tsv",
        sep="\t"
    )

    source2 = pd.read_csv(
        TEST_DIR / "test_source2.tsv",
        sep="\t"
    )

    source3 = pd.read_csv(
        TEST_DIR / "test_source3.tsv",
        sep="\t"
    )

    return source1, source2, source3


if __name__ == "__main__":

    train_s1, train_s2, train_s3, ground_truth = load_training_data()

    test_s1, test_s2, test_s3 = load_test_data()

    print("TRAINING DATA")
    print("Source 1:", train_s1.shape)
    print("Source 2:", train_s2.shape)
    print("Source 3:", train_s3.shape)
    print("Ground Truth:", ground_truth.shape)

    print("\nTEST DATA")
    print("Source 1:", test_s1.shape)
    print("Source 2:", test_s2.shape)
    print("Source 3:", test_s3.shape)