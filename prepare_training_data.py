import pandas as pd

from sklearn.model_selection import train_test_split


INPUT_PATH = "data/emotion_queries.csv"

TRAIN_PATH = "data/train.csv"
VAL_PATH = "data/validation.csv"
TEST_PATH = "data/test.csv"


def main():

    df = pd.read_csv(INPUT_PATH)

    print("Total samples:", len(df))

    # 70% training, 30% temporary
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        stratify=df["emotion"],
        random_state=42,
    )

    # Split remaining 30% equally:
    # 15% validation, 15% test
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df["emotion"],
        random_state=42,
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    val_df.to_csv(VAL_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("\nDataset split:")
    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Test:", len(test_df))

    print("\nTraining distribution:")
    print(train_df["emotion"].value_counts())

    print("\nValidation distribution:")
    print(val_df["emotion"].value_counts())

    print("\nTest distribution:")
    print(test_df["emotion"].value_counts())

    print("\nFiles created successfully.")


if __name__ == "__main__":
    main()