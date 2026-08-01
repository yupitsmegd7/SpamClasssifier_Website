"""
Trains the spam/ham text classifier — same logic as spammer.ipynb, just as a
plain script so it can be re-run outside of Jupyter/Colab.

Usage:
    python train_model.py path/to/SMSSpamCollection.tsv
"""

import sys
import random

import pandas as pd
import spacy
from spacy.training import Example


def main(data_path):
    # ---- 1. Load + clean data (same steps as the notebook) ----
    data = pd.read_csv(data_path, sep="\t", header=None, names=["label", "message"])
    data.rename(columns={"label": "target", "message": "write"}, inplace=True)
    data["target"] = data["target"].apply(lambda x: 1 if x == "spam" else 0)
    data = data.drop_duplicates(keep="first")

    # ---- 2. Build spaCy training examples ----
    # NOTE: this keeps the notebook's original (slightly confusing) label
    # mapping — REAL = spam, FAKE = ham — so the trained model's output
    # format matches what the notebook produced.
    data["refining"] = data.apply(
        lambda row: (
            row["write"],
            {"cats": {"FAKE": float(1 - row["target"]), "REAL": float(row["target"])}},
        ),
        axis=1,
    )

    nlp = spacy.blank("en")
    textcat = nlp.add_pipe("textcat")
    textcat.add_label("FAKE")
    textcat.add_label("REAL")

    train_data = data["refining"].tolist()

    examples = []
    for text, annotations in train_data:
        doc = nlp.make_doc(text)
        example = Example.from_dict(doc, annotations)
        examples.append(example)

    # ---- 3. Train ----
    nlp.initialize(lambda: examples)
    optimizer = nlp.create_optimizer()

    for epoch in range(10):
        random.shuffle(examples)
        losses = {}
        for batch in spacy.util.minibatch(examples, size=10):
            nlp.update(batch, sgd=optimizer, losses=losses)
        print(f"Epoch {epoch}: Loss = {losses['textcat']:.4f}")

    # ---- 4. Save ----
    nlp.to_disk("./fake_review_model")
    print("Saved model to ./fake_review_model")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "SMSSpamCollection_expanded.tsv"
    main(path)
