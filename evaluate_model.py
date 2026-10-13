import pandas as pd
import numpy as np
import json

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

MODEL_PATH = "models/distilbert-emotion"
TEST_PATH = "data/test.csv"


def main():

    print("=" * 60)
    print("SHLOKASENSE - MODEL EVALUATION")
    print("=" * 60)

    df = pd.read_csv(TEST_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH
    )

    label2id = model.config.label2id
    id2label = {
        int(k): v for k, v in model.config.id2label.items()
    }

    df["label"] = df["emotion"].map(label2id)

    dataset = Dataset.from_pandas(
        df[["text", "label"]],
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    dataset = dataset.map(
        tokenize,
        batched=True,
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
    )

    predictions = trainer.predict(dataset)

    y_true = predictions.label_ids
    y_pred = np.argmax(
        predictions.predictions,
        axis=-1,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    print("\nTEST RESULTS")
    print("-" * 40)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("\nCLASSIFICATION REPORT")
    print("-" * 40)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                id2label[i]
                for i in range(len(id2label))
            ],
            zero_division=0,
        )
    )

    print("CONFUSION MATRIX")
    print("-" * 40)
    print(confusion_matrix(y_true, y_pred))

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }

    with open(
        "models/distilbert-emotion/metrics.json",
        "w",
    ) as f:
        json.dump(metrics, f, indent=4)

    print(
        "\nMetrics saved to "
        "models/distilbert-emotion/metrics.json"
    )


if __name__ == "__main__":
    main()