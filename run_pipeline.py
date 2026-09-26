from src.data_loader import load_training_data, load_test_data


def main():
    print("Amazon ML Challenge 2026")
    print("Business Entity Resolution Pipeline")
    print("------------------------------------")

    # 1. Load data
    print("1. Loading data...")

    train_s1, train_s2, train_s3, ground_truth = load_training_data()
    test_s1, test_s2, test_s3 = load_test_data()

    print(f"   Train Source 1: {len(train_s1):,} records")
    print(f"   Train Source 2: {len(train_s2):,} records")
    print(f"   Train Source 3: {len(train_s3):,} records")
    print(f"   Ground Truth:   {len(ground_truth):,} records")

    print(f"   Test Source 1:  {len(test_s1):,} records")
    print(f"   Test Source 2:  {len(test_s2):,} records")
    print(f"   Test Source 3:  {len(test_s3):,} records")

    # 2. Preprocessing
    print("2. Preprocessing...")
    # Member 3 will connect their preprocessing module here.

    # 3. Candidate generation
    print("3. Candidate generation...")
    # Member 4 will connect blocking here.

    # 4. Feature engineering
    print("4. Feature engineering...")
    # Member 4 will connect feature engineering here.

    # 5. Model inference
    print("5. Model inference...")
    # Member 5 will connect the trained model here.

    # 6. Generate outputs
    print("6. Generating outputs...")
    # Final matching_results.tsv will be generated here.

    print("------------------------------------")
    print("Pipeline structure loaded successfully!")


if __name__ == "__main__":
    main()