"""
IMDB Sentiment Analysis - Data Preprocessing

Responsibilities:
1. Load dataset
2. Clean reviews
3. Encode labels
4. Train/Test Split
5. Tokenization
6. Sequence Conversion
7. Padding
8. Save tokenizer
"""

import re
import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==================================================
# CONFIG
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "IMDB_Dataset.csv"
MODELS_DIR = BASE_DIR / "models"

NUM_WORDS = 10000
MAX_LEN = 300
TEST_SIZE = 0.2
RANDOM_STATE = 42


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text):
    """
    Clean review text.

    Steps:
    1. Fix escaped apostrophes
    2. Remove HTML tags
    3. Convert to lowercase
    4. Remove extra spaces
    """

    text = text.replace("\\'", "'")

    text = re.sub(r"<.*?>", " ", text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==================================================
# LOAD DATASET
# ==================================================

def load_dataset():
    """
    Load IMDB dataset and clean reviews.
    """

    df = pd.read_csv(DATA_PATH)

    df["review"] = df["review"].apply(clean_text)

    return df


# ==================================================
# LABEL ENCODING
# ==================================================

def encode_labels(labels):
    """
    positive -> 1
    negative -> 0
    """

    return labels.map({
        "positive": 1,
        "negative": 0
    })


# ==================================================
# TOKENIZER
# ==================================================

def create_tokenizer():
    """
    Create tokenizer.
    """

    tokenizer = Tokenizer(
        num_words=NUM_WORDS,
        oov_token="<OOV>"
    )

    return tokenizer


# ==================================================
# SAVE / LOAD TOKENIZER
# ==================================================

def save_tokenizer(tokenizer):
    """
    Save tokenizer for deployment.
    """

    MODELS_DIR.mkdir(exist_ok=True)

    tokenizer_path = MODELS_DIR / "tokenizer.pkl"

    with open(tokenizer_path, "wb") as file:
        pickle.dump(tokenizer, file)

    print(f"Tokenizer saved: {tokenizer_path}")


def load_tokenizer():
    """
    Load tokenizer.
    """

    tokenizer_path = MODELS_DIR / "tokenizer.pkl"

    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    return tokenizer


# ==================================================
# MAIN PREPROCESSING PIPELINE
# ==================================================

def prepare_training_data():
    """
    Complete preprocessing pipeline.
    """

    print("Loading dataset...")

    df = load_dataset()

    X = df["review"]
    y = encode_labels(df["sentiment"])

    print("Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("Creating tokenizer...")

    tokenizer = create_tokenizer()

    tokenizer.fit_on_texts(X_train)

    print("Converting text to sequences...")

    train_sequences = tokenizer.texts_to_sequences(X_train)
    test_sequences = tokenizer.texts_to_sequences(X_test)

    print("Padding sequences...")

    X_train_padded = pad_sequences(
        train_sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    X_test_padded = pad_sequences(
        test_sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post"
    )

    save_tokenizer(tokenizer)

    print("\nPreprocessing Complete")
    print(f"Train Shape: {X_train_padded.shape}")
    print(f"Test Shape : {X_test_padded.shape}")

    return (
        X_train_padded,
        X_test_padded,
        y_train.values,
        y_test.values,
        tokenizer
    )


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer
    ) = prepare_training_data()