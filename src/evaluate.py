"""
IMDB Sentiment Analysis - Model Evaluation

Responsibilities:
1. Load trained model
2. Load training history
3. Evaluate test performance
4. Generate graphs
5. Generate confusion matrix
6. Generate classification report
7. Generate prediction distribution
"""

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from preprocess import prepare_training_data


# ==================================================
# CONFIG
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

MODEL_PATH = MODELS_DIR / "best_model.keras"
HISTORY_PATH = MODELS_DIR / "history.pkl"

REPORT_PATH = SCREENSHOTS_DIR / "classification_report.txt"


# ==================================================
# LOAD ARTIFACTS
# ==================================================

def load_trained_model():
    """
    Load saved keras model.
    """

    model = load_model(
        MODEL_PATH
    )

    return model


def load_history():
    """
    Load training history.
    """

    with open(
        HISTORY_PATH,
        "rb"
    ) as file:

        history = pickle.load(file)

    return history


# ==================================================
# TRAINING CURVES
# ==================================================

def plot_accuracy(history):
    """
    Plot accuracy curve.
    """

    epochs = range(
        1,
        len(history["accuracy"]) + 1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(
        "Training vs Validation Accuracy"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid()

    plt.savefig(
        SCREENSHOTS_DIR / "accuracy_curve.png",
        bbox_inches="tight"
    )

    plt.close()


def plot_loss(history):
    """
    Plot loss curve.
    """

    epochs = range(
        1,
        len(history["loss"]) + 1
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["loss"],
        label="Training Loss"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "Training vs Validation Loss"
    )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid()

    plt.savefig(
        SCREENSHOTS_DIR / "loss_curve.png",
        bbox_inches="tight"
    )

    plt.close()


# ==================================================
# PREDICTIONS
# ==================================================

def generate_predictions(
    model,
    X_test
):
    """
    Generate probabilities and classes.
    """

    probabilities = model.predict(
        X_test,
        verbose=0
    )

    predictions = (
        probabilities > 0.5
    ).astype("int32")

    return probabilities, predictions


# ==================================================
# METRICS
# ==================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate test data.
    """

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(
        f"Test Loss: {loss:.4f}"
    )

    print(
        f"Test Accuracy: {accuracy:.4f}"
    )


def save_classification_report(
    y_test,
    predictions
):
    """
    Save classification report.
    """

    report = classification_report(
        y_test,
        predictions,
        target_names=[
            "Negative",
            "Positive"
        ]
    )

    print(report)

    with open(
        REPORT_PATH,
        "w"
    ) as file:

        file.write(report)


def plot_confusion_matrix(
    y_test,
    predictions
):
    """
    Generate confusion matrix.
    """

    cm = confusion_matrix(
        y_test,
        predictions
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Negative",
            "Positive"
        ]
    )

    display.plot()

    plt.title(
        "Confusion Matrix"
    )

    plt.savefig(
        SCREENSHOTS_DIR / "confusion_matrix.png",
        bbox_inches="tight"
    )

    plt.close()


def plot_prediction_distribution(
    probabilities
):
    """
    Plot prediction confidence distribution.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        probabilities,
        bins=50
    )

    plt.title(
        "Prediction Probability Distribution"
    )

    plt.xlabel(
        "Probability"
    )

    plt.ylabel(
        "Count"
    )

    plt.grid()

    plt.savefig(
        SCREENSHOTS_DIR / "prediction_distribution.png",
        bbox_inches="tight"
    )

    plt.close()


# ==================================================
# MAIN
# ==================================================

def main():

    SCREENSHOTS_DIR.mkdir(
        exist_ok=True
    )


    (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer
    ) = prepare_training_data()


    model = load_trained_model()

    history = load_history()


    plot_accuracy(history)

    plot_loss(history)


    evaluate_model(
        model,
        X_test,
        y_test
    )


    probabilities, predictions = generate_predictions(
        model,
        X_test
    )


    save_classification_report(
        y_test,
        predictions
    )


    plot_confusion_matrix(
        y_test,
        predictions
    )


    plot_prediction_distribution(
        probabilities
    )


    print(
        "\nEvaluation completed."
    )


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    main()