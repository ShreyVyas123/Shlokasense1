import pandas as pd

INPUT_PATH = "data/bhagavad_gita.csv"
OUTPUT_PATH = "data/gita_clean.csv"


def main():
    # Load dataset
    df = pd.read_csv(INPUT_PATH)

    print("=" * 60)
    print("SHLOKASENSE - DATA CLEANING")
    print("=" * 60)

    print("\nOriginal shape:", df.shape)

    # Remove unnecessary index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Remove exact duplicate rows
    before = len(df)
    df = df.drop_duplicates()
    print("Duplicate rows removed:", before - len(df))

    # Required fields must not be missing
    required_columns = [
        "verse_number",
        "verse_in_sanskrit",
        "translation_in_english",
        "meaning_in_english",
    ]

    before = len(df)
    df = df.dropna(subset=required_columns)
    print("Rows with missing required fields removed:", before - len(df))

    # Fill optional Hindi field
    df["meaning_in_hindi"] = df["meaning_in_hindi"].fillna("")

    # Clean whitespace
    text_columns = [
        "verse_number",
        "verse_in_sanskrit",
        "sanskrit_verse_transliteration",
        "translation_in_english",
        "meaning_in_english",
        "translation_in_hindi",
        "meaning_in_hindi",
    ]

    for column in text_columns:
        df[column] = df[column].astype(str).str.strip()

    # Text that will later be used for semantic retrieval
    df["retrieval_text"] = (
        df["translation_in_english"]
        + " "
        + df["meaning_in_english"]
    )

    # Save cleaned dataset
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaned shape:", df.shape)
    print("Saved to:", OUTPUT_PATH)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nFinal columns:")
    print(df.columns.tolist())

    print("\nDataset cleaning completed successfully!")


if __name__ == "__main__":
    main()