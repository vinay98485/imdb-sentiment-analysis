"""
IMDB Sentiment Analysis - Model Training

Responsibilities:
1. Load processed data
2. Build LSTM model
3. Configure callbacks
4. Train model
5. Save training history
6. Evaluate model
"""

import pickle
from pathlib import Path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from preprocess import prepare_training_data


# ==================================================
# CONFIG
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODELS_DIR = BASE_DIR / "models"

MODEL_PATH = MODELS_DIR / "best_model.keras"
HISTORY_PATH = MODELS_DIR / "history.pkl"

VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
LSTM_UNITS = 128

EPOCHS = 20
BATCH_SIZE = 32


# ==================================================
# MODEL
# ==================================================

def build_model():
    """
    Build LSTM sentiment classifier.
    """

    model = Sequential([
        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM,
            mask_zero=True
        ),

        LSTM(LSTM_UNITS),

        Dropout(0.5),

        Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ==================================================
# CALLBACKS
# ==================================================

def get_callbacks():
    """
    Create training callbacks.
    """

    MODELS_DIR.mkdir(exist_ok=True)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=MODEL_PATH,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1
    )

    return [
        early_stopping,
        checkpoint,
        reduce_lr
    ]


# ==================================================
# SAVE HISTORY
# ==================================================

def save_history(history):
    """
    Save training history.
    """

    with open(HISTORY_PATH, "wb") as file:
        pickle.dump(
            history.history,
            file
        )

    print(f"History saved: {HISTORY_PATH}")


# ==================================================
# TRAINING
# ==================================================

def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Train model.
    """

    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(
            X_test,
            y_test
        ),
        callbacks=get_callbacks(),
        verbose=1
    )

    return history


# ==================================================
# EVALUATION
# ==================================================

def evaluate_model(
    model,
    X_test,
    y_test
):
    """
    Evaluate model.
    """

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


# ==================================================
# MAIN
# ==================================================

def main():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        tokenizer
    ) = prepare_training_data()


    print("\nBuilding model...")

    model = build_model()

    model.summary()


    print("\nTraining model...")

    history = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test
    )


    save_history(history)


    evaluate_model(
        model,
        X_test,
        y_test
    )


    print("\nTraining completed.")


# ==================================================
# TEST
# ==================================================

if __name__ == "__main__":

    main()