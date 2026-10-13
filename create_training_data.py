import random
import pandas as pd

OUTPUT_PATH = "data/emotion_queries.csv"

random.seed(42)

TEMPLATES = {
    "anxiety": [
        "I feel anxious about {topic}.",
        "I am worried about {topic}.",
        "I cannot stop worrying about {topic}.",
        "I keep overthinking {topic}.",
        "I feel nervous about {topic}.",
        "I am constantly stressed about {topic}.",
        "I am afraid something will go wrong with {topic}.",
        "I feel restless whenever I think about {topic}.",
        "My mind is full of worries about {topic}.",
        "I don't know how to deal with my anxiety about {topic}.",
    ],

    "anger": [
        "I am extremely angry about {topic}.",
        "I cannot control my anger about {topic}.",
        "I feel furious because of {topic}.",
        "I am frustrated and angry about {topic}.",
        "I feel rage whenever I think about {topic}.",
        "I am struggling to let go of my anger about {topic}.",
        "What happened with {topic} made me very angry.",
        "I feel irritated whenever I deal with {topic}.",
        "I cannot stop being angry about {topic}.",
        "I feel resentment because of {topic}.",
    ],

    "grief": [
        "I am grieving because of {topic}.",
        "I am struggling with the loss of {topic}.",
        "I cannot get over losing {topic}.",
        "I feel deep grief because of {topic}.",
        "I am mourning the loss of {topic}.",
        "The loss of {topic} has broken me emotionally.",
        "I don't know how to deal with my grief over {topic}.",
        "I feel devastated after losing {topic}.",
        "I am finding it difficult to accept the loss of {topic}.",
        "I keep thinking about the person I lost because of {topic}.",
    ],

    "confusion": [
        "I feel completely confused about {topic}.",
        "I don't know what to do about {topic}.",
        "I feel lost when I think about {topic}.",
        "I cannot decide what to do about {topic}.",
        "I am unsure about the right decision regarding {topic}.",
        "I don't understand what I should do about {topic}.",
        "I feel stuck and confused about {topic}.",
        "I need clarity about {topic}.",
        "My mind is confused because of {topic}.",
        "I am struggling to understand what is right regarding {topic}.",
    ],

    "fear": [
        "I am afraid of {topic}.",
        "I feel scared about {topic}.",
        "I am terrified about {topic}.",
        "I fear that {topic} will go wrong.",
        "I feel frightened whenever I think about {topic}.",
        "I don't know how to overcome my fear of {topic}.",
        "The thought of {topic} scares me.",
        "I am worried that {topic} might happen.",
        "I feel unsafe because of {topic}.",
        "My fear about {topic} is overwhelming me.",
    ],

    "sadness": [
        "I feel very sad about {topic}.",
        "I have been feeling unhappy because of {topic}.",
        "I feel emotionally low because of {topic}.",
        "I don't feel happy anymore because of {topic}.",
        "I feel hopeless about {topic}.",
        "I am going through a very difficult time with {topic}.",
        "I feel emotionally exhausted because of {topic}.",
        "Nothing seems to make me happy because of {topic}.",
        "I feel empty when I think about {topic}.",
        "I am struggling with sadness related to {topic}.",
    ],

    "peace": [
        "I want to find peace regarding {topic}.",
        "I want to feel calm about {topic}.",
        "I want inner peace despite {topic}.",
        "I want to stop worrying and become peaceful about {topic}.",
        "I want to accept {topic} peacefully.",
        "I want to develop a calm mind regarding {topic}.",
        "I want emotional balance when dealing with {topic}.",
        "I want to remain peaceful despite {topic}.",
        "I want to overcome mental disturbance caused by {topic}.",
        "I want serenity in my life despite {topic}.",
    ],

    "motivation": [
        "I need motivation to deal with {topic}.",
        "I need strength to continue with {topic}.",
        "I want motivation to keep going with {topic}.",
        "I need courage to face {topic}.",
        "I want to stay focused on {topic}.",
        "I need encouragement because of {topic}.",
        "I want the strength to overcome {topic}.",
        "I need guidance to keep working toward {topic}.",
        "I want to become stronger while dealing with {topic}.",
        "I need determination to continue despite {topic}.",
    ],

    "joy": [
        "I am so happy about {topic}.",
        "I feel joyful because of {topic}.",
        "I am really happy with {topic}.",
        "I feel great about {topic}.",
        "I am proud of {topic}.",
        "I feel grateful for {topic}.",
        "I am excited about {topic}.",
        "I feel content with {topic}.",
        "Everything is going well with {topic}.",
        "I feel fulfilled because of {topic}.",
    ],
}


TOPICS = [
    "my career",
    "my studies",
    "my exams",
    "my future",
    "my relationships",
    "my family",
    "my responsibilities",
    "an important decision",
    "a difficult situation",
    "my personal problems",
    "my failures",
    "my goals",
    "my work",
    "a major change in my life",
    "my expectations",
    "my friendships",
    "my financial situation",
    "my personal growth",
    "my responsibilities",
    "an uncertain future",
]


def create_dataset(samples_per_emotion=200):
    rows = []

    for emotion, templates in TEMPLATES.items():
        for _ in range(samples_per_emotion):
            template = random.choice(templates)
            topic = random.choice(TOPICS)

            text = template.format(topic=topic)

            rows.append(
                {
                    "text": text,
                    "emotion": emotion,
                }
            )

    random.shuffle(rows)

    return pd.DataFrame(rows)


def main():
    df = create_dataset(samples_per_emotion=200)

    df.to_csv(OUTPUT_PATH, index=False)

    print("=" * 60)
    print("SHLOKASENSE - EMOTION TRAINING DATASET")
    print("=" * 60)

    print("\nDataset shape:")
    print(df.shape)

    print("\nEmotion distribution:")
    print(df["emotion"].value_counts())

    print("\nSample records:")
    print(df.head(15).to_string(index=False))

    print("\nSaved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()