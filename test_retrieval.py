import pickle

import faiss
import numpy as np

from sentence_transformers import SentenceTransformer


INDEX_PATH = "models/gita_faiss.index"
METADATA_PATH = "models/gita_metadata.pkl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def search(query, top_k=5):

    model = SentenceTransformer(
        MODEL_NAME
    )

    index = faiss.read_index(
        INDEX_PATH
    )

    with open(
        METADATA_PATH,
        "rb",
    ) as f:
        metadata = pickle.load(f)

    embedding = model.encode(
        [query],
        normalize_embeddings=True,
    )

    scores, indices = index.search(
        embedding.astype("float32"),
        top_k,
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0],
    ):
        result = metadata[index_id].copy()
        result["similarity"] = float(score)
        results.append(result)

    return results


if __name__ == "__main__":

    query = input(
        "\nDescribe your problem: "
    )

    results = search(query)

    print("\n" + "=" * 60)
    print("RELEVANT BHAGAVAD GITA VERSES")
    print("=" * 60)

    for i, result in enumerate(
        results,
        start=1,
    ):

        print(
            f"\n{i}. {result['verse_number']}"
        )

        print(
            f"Similarity: "
            f"{result['similarity']:.4f}"
        )

        print(
            "\nTranslation:"
        )

        print(
            result[
                "translation_in_english"
            ]
        )

        print(
            "\nMeaning:"
        )

        print(
            result[
                "meaning_in_english"
            ]
        )