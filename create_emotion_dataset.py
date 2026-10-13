import pandas as pd

INPUT_PATH = "data/gita_clean.csv"
OUTPUT_PATH = "data/gita_emotions.csv"


EMOTION_KEYWORDS = {
    "anxiety": [
        "worry",
        "worried",
        "anxiety",
        "anxious",
        "distress",
        "uncertain",
        "uncertainty",
        "concern",
        "restless",
        "agitated",
    ],
    "anger": [
        "anger",
        "angry",
        "wrath",
        "rage",
        "hatred",
        "hostility",
        "resentment",
    ],
    "grief": [
        "grief",
        "grieve",
        "mourning",
        "loss",
        "lament",
        "sorrow",
        "suffering",
        "despair",
    ],
    "confusion": [
        "confused",
        "confusion",
        "doubt",
        "doubtful",
        "delusion",
        "illusion",
        "ignorance",
        "uncertain",
    ],
    "fear": [
        "fear",
        "afraid",
        "fright",
        "terrified",
        "danger",
        "threat",
    ],
    "sadness": [
        "sad",
        "sadness",
        "sorrow",
        "pain",
        "unhappy",
        "misery",
        "suffering",
    ],
    "peace": [
        "peace",
        "peaceful",
        "calm",
        "tranquil",
        "serenity",
        "equanimity",
        "contentment",
        "harmony",
    ],
    "motivation": [
        "duty",
        "action",
        "work",
        "discipline",
        "courage",
        "strength",
        "purpose",
        "knowledge",
        "wisdom",
        "devotion",
    ],
    "joy": [
        "happy",
        "happiness",
        "joy",
        "joyful",
        "delight",
        "bliss",
        "gladness",
        "cheerful",
        "content",
        "contentment",
        "fulfilled",
        "grateful",
        "gratitude",
        "proud",
        "excited",
    ],
}


def assign_emotion(text):
    text = text.lower()

    scores = {}

    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(
            1 for keyword in keywords
            if keyword in text
        )
        scores[emotion] = score

    best_emotion = max(scores, key=scores.get)

    # If no keyword matches, use motivation as a
    # temporary neutral/default category.
    if scores[best_emotion] == 0:
        return "motivation"

    return best_emotion


def main():
    df = pd.read_csv(INPUT_PATH)

    # Use English meaning + translation as the
    # semantic source for the initial annotation.
    df["emotion_text"] = (
        df["translation_in_english"]
        + " "
        + df["meaning_in_english"]
    )

    df["emotion"] = df["emotion_text"].apply(assign_emotion)

    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)
    print("SHLOKASENSE - INITIAL EMOTION ANNOTATION")
    print("=" * 60)

    print("\nDataset shape:", df.shape)

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts())

    print("\nEmotion percentages:")
    print(
        (df["emotion"].value_counts(normalize=True) * 100)
        .round(2)
    )

    print("\nSample labelled records:")

    print(
        df[
            [
                "verse_number",
                "emotion",
                "translation_in_english",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()