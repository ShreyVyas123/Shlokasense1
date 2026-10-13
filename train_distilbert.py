import os

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "models/distilbert-emotion"


EMOTIONS = [
    "anxiety",
    "anger",
    "grief",
    "confusion",
    "fear",
    "sadness",
    "peace",
    "motivation",
    "joy",
]


label2id = {
    emotion: index
    for index, emotion in enumerate(EMOTIONS)
}

id2label = {
    index: emotion
    for emotion, index in label2id.items()
}


def load_data(path):

    df = pd.read_csv(path)

    df["label"] = df["emotion"].map(label2id)

    return Dataset.from_pandas(
        df[["text", "label"]],
        preserve_index=False,
    )


def compute_metrics(eval_prediction):

    logits, labels = eval_prediction

    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():

    print("=" * 60)
    print("SHLOKASENSE - DISTILBERT EMOTION CLASSIFIER")
    print("=" * 60)

    train_dataset = load_data(
        "data/train.csv"
    )

    validation_dataset = load_data(
        "data/validation.csv"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=128,
        )

    train_dataset = train_dataset.map(
        tokenize,
        batched=True,
    )

    validation_dataset = validation_dataset.map(
        tokenize,
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(EMOTIONS),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("\nStarting training...\n")

    trainer.train()

    print("\nFinal validation metrics:")

    metrics = trainer.evaluate()

    for key, value in metrics.items():
        print(f"{key}: {value}")

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    trainer.save_model(
        OUTPUT_DIR
    )

    tokenizer.save_pretrained(
        OUTPUT_DIR
    )

    print("\nModel saved to:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()