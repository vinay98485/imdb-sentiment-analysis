"""
IMDB Sentiment Analysis - Prediction

Responsibilities:
1. Load trained model
2. Load tokenizer
3. Clean user input
4. Convert text into sequence
5. Predict sentiment
"""

import pickle
from pathlib import Path

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.preprocess import clean_text


# ==================================================
# CONFIG
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "best_model.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"

MAX_LEN = 300


# ==================================================
# LOAD ARTIFACTS
# ==================================================

def load_trained_model():
    """
    Load trained keras model.
    """

    model = load_model(
        MODEL_PATH
    )

    return model


def load_tokenizer():
    """
    Load saved tokenizer.
    """

    with open(
        TOKENIZER_PATH,
        "rb"
    ) as file:

        tokenizer = pickle.load(file)

    return tokenizer


# ==================================================
# PREPROCESS INPUT
# ==================================================

def preprocess_input(
    text,
    tokenizer
):
    """
    Convert text into padded sequence.
    """

    text = clean_text(
        text
    )

    sequence = tokenizer.texts_to_sequences(
        [text]
    )

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    return padded_sequence


# ==================================================
# PREDICTION
# ==================================================

def predict_sentiment(
    text
):
    """
    Predict movie review sentiment.
    """

    model = load_trained_model()

    tokenizer = load_tokenizer()


    processed_text = preprocess_input(
        text,
        tokenizer
    )


    probability = model.predict(
        processed_text,
        verbose=0
    )[0][0]


    if probability >= 0.5:

        sentiment = "Positive"

    else:

        sentiment = "Negative"


    return sentiment, probability


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    review = input(
        "\nEnter movie review: "
    )


    sentiment, probability = predict_sentiment(
        review
    )


    print(
        f"\nSentiment: {sentiment}"
    )

    print(
        f"Probability: {probability:.4f}"
    )