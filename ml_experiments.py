# ml_experiments.py
"""
Simple ML experiments for the Mood Machine lab.

This file uses a "real" machine learning library (scikit-learn)
to train a tiny text classifier on the same SAMPLE_POSTS and
TRUE_LABELS that you use with the rule based model.
"""

from typing import List, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from dataset import (
    BASELINE_POSTS,
    BASELINE_LABELS,
    PART4_POSTS,
    PART4_LABELS,
    SAMPLE_POSTS,
    TRUE_LABELS,
)
from mood_analyzer import MoodAnalyzer


def train_ml_model(
    texts: List[str],
    labels: List[str],
) -> Tuple[CountVectorizer, LogisticRegression]:
    """
    Train a simple text classifier using bag of words features
    and logistic regression.

    Steps:
      1. Convert the texts into numeric vectors using CountVectorizer.
      2. Fit a LogisticRegression model on those vectors and labels.

    Returns:
      (vectorizer, model)
    """
    if len(texts) != len(labels):
        raise ValueError(
            "texts and labels must be the same length. "
            "Check SAMPLE_POSTS and TRUE_LABELS in dataset.py."
        )

    if not texts:
        raise ValueError("No training data provided. Add examples in dataset.py.")

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    return vectorizer, model


def evaluate_on_dataset(
    texts: List[str],
    labels: List[str],
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> float:
    """
    Evaluate the trained model on a labeled dataset.

    Prints each text with its predicted label and the true label,
    then returns the overall accuracy as a float between 0 and 1.
    """
    if len(texts) != len(labels):
        raise ValueError(
            "texts and labels must be the same length. "
            "Check your dataset."
        )

    X = vectorizer.transform(texts)
    preds = model.predict(X)

    print("=== ML Model Evaluation on Dataset ===")
    correct = 0
    for text, true_label, pred_label in zip(texts, labels, preds):
        is_correct = pred_label == true_label
        if is_correct:
            correct += 1
        print(f'"{text}" -> predicted={pred_label}, true={true_label}')

    accuracy = accuracy_score(labels, preds)
    print(f"\nAccuracy on this dataset: {accuracy:.2f}")
    return accuracy


def predict_single_text(
    text: str,
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> str:
    """
    Predict the mood label for a single text string using
    the trained ML model.
    """
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    return pred


def run_interactive_loop(
    vectorizer: CountVectorizer,
    model: LogisticRegression,
) -> None:
    """
    Let the user type their own sentences and see the ML model's
    predicted mood label.

    Type 'quit' or press Enter on an empty line to exit.
    """
    print("\n=== Interactive Mood Machine (ML model) ===")
    print("Type a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the ML Mood Machine.")
            break

        label = predict_single_text(user_input, vectorizer, model)
        print(f"ML model: {label}")


def run_data_expansion_experiment() -> None:
    """Compare fixed rules with ML predictions before and after adding data."""
    before_vectorizer, before_model = train_ml_model(BASELINE_POSTS, BASELINE_LABELS)
    after_vectorizer, after_model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)
    rule = MoodAnalyzer()

    before_predictions = before_model.predict(before_vectorizer.transform(PART4_POSTS))
    after_predictions = after_model.predict(after_vectorizer.transform(PART4_POSTS))
    rule_predictions = [rule.predict_label(text) for text in PART4_POSTS]

    print("=== Part 4: Data Expansion Experiment ===")
    print(f"Baseline: {len(BASELINE_POSTS)} posts; expanded: {len(SAMPLE_POSTS)} posts")
    print("Word lists and scoring rules are unchanged in this comparison.")
    for text, label, rule_label, before_label, after_label in zip(
        PART4_POSTS, PART4_LABELS, rule_predictions, before_predictions, after_predictions
    ):
        print(
            f'"{text}" -> true={label}, rule={rule_label}, '
            f'ML before={before_label}, ML after={after_label}'
        )

    print(f"Rules on the 3 added posts: {accuracy_score(PART4_LABELS, rule_predictions):.2f}")
    print(
        "ML before, 3 unseen posts: "
        f"{accuracy_score(PART4_LABELS, before_predictions):.2f}"
    )
    print(
        "ML after, same 3 posts now in training: "
        f"{accuracy_score(PART4_LABELS, after_predictions):.2f}"
    )
    print("The after score reuses training examples; it is not an independent test score.")
    print("The tiny before sample is a learning exercise, not a generalization benchmark.\n")


if __name__ == "__main__":
    run_data_expansion_experiment()

    print("Training an ML model on SAMPLE_POSTS and TRUE_LABELS from dataset.py...")
    print("Make sure you have added enough labeled examples before running this.\n")

    # Train the model on the current dataset.
    vectorizer, model = train_ml_model(SAMPLE_POSTS, TRUE_LABELS)

    # Evaluate on the same dataset (training accuracy).
    evaluate_on_dataset(SAMPLE_POSTS, TRUE_LABELS, vectorizer, model)

    # Let the user try their own examples.
    run_interactive_loop(vectorizer, model)

    print("\nTip: Compare these predictions with the rule based model")
    print("by running `python main.py`. Notice where they fail in")
    print("similar ways and where they fail in different ways.")
