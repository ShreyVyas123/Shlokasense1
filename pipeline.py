import pickle

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)


EMOTION_MODEL = (
    "models/distilbert-emotion"
)

FAISS_INDEX = (
    "models/gita_faiss.index"
)

METADATA = (
    "models/gita_metadata.pkl"
)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


class ShlokaSensePipeline:

    def __init__(self):

        print("Loading emotion model...")

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                EMOTION_MODEL
            )
        )

        self.emotion_model = (
            AutoModelForSequenceClassification
            .from_pretrained(
                EMOTION_MODEL
            )
        )

        print("Loading Sentence-BERT...")

        self.embedding_model = (
            SentenceTransformer(
                EMBEDDING_MODEL
            )
        )

        print("Loading FAISS...")

        self.index = faiss.read_index(
            FAISS_INDEX
        )

        with open(
            METADATA,
            "rb",
        ) as f:
            self.metadata = pickle.load(f)

        print("Pipeline ready!")

    def detect_emotion(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        outputs = (
            self.emotion_model(
                **inputs
            )
        )

        probabilities = (
            outputs.logits.softmax(
                dim=-1
            )
        )

        prediction = (
            probabilities.argmax(
                dim=-1
            ).item()
        )

        emotion = (
            self.emotion_model
            .config
            .id2label[prediction]
        )

        confidence = (
            probabilities[
                0,
                prediction
            ].item()
        )

        return emotion, confidence

    def retrieve(
        self,
        query,
        top_k=3,
    ):

        embedding = (
            self.embedding_model.encode(
                [query],
                normalize_embeddings=True,
            )
        )

        scores, indices = (
            self.index.search(
                embedding.astype(
                    "float32"
                ),
                top_k,
            )
        )

        results = []

        for score, index_id in zip(
            scores[0],
            indices[0],
        ):

            result = self.metadata[
                index_id
            ].copy()

            result[
                "similarity"
            ] = float(score)

            results.append(
                result
            )

        return results

    def predict(
        self,
        query,
        top_k=3,
    ):

        emotion, confidence = (
            self.detect_emotion(
                query
            )
        )

        verses = self.retrieve(
            query,
            top_k,
        )

        return {
            "query": query,
            "emotion": emotion,
            "confidence": confidence,
            "verses": verses,
        }


if __name__ == "__main__":

    pipeline = (
        ShlokaSensePipeline()
    )

    query = input(
        "\nHow are you feeling? "
    )

    result = pipeline.predict(
        query
    )

    print("\nDetected emotion:")
    print(
        result["emotion"]
    )

    print(
        "\nConfidence:",
        round(
            result["confidence"],
            4,
        ),
    )

    for i, verse in enumerate(
        result["verses"],
        start=1,
    ):

        print(
            f"\n{i}. "
            f"{verse['verse_number']}"
        )

        print(
            verse[
                "translation_in_english"
            ]
        )