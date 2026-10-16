"""
Training script for the motion-based gesture project.

Architecture based on the Edge Impulse screenshot:
39 input features -> Dense(20) -> Dense(10) -> 4-class softmax
Learning rate: 0.0005

IMPORTANT:
This script expects a CSV containing ALREADY-EXTRACTED 39 model features
and a 'label' column. It does not recreate Edge Impulse's Spectral Features
processing from raw AX/AY/AZ data. Export/prepare features that match the
same Edge Impulse impulse before using this script.

Classes: FOOD, HELP, MEDICINE, WATER
The achieved accuracy depends on the data and split; 85% cannot be guaranteed.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

SEED = 42
FEATURE_COUNT = 39
CLASSES = ["FOOD", "HELP", "MEDICINE", "WATER"]
LEARNING_RATE = 0.0005
DATA_FILE = Path(__file__).resolve().parent / "training_features.csv"
MODEL_DIR = Path(__file__).resolve().parent / "output"

np.random.seed(SEED)
tf.random.set_seed(SEED)


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"\nMissing {DATA_FILE.name}.\n"
            "Create this CSV with 39 feature columns named f0..f38 and a 'label' column.\n"
            "The features must match the features produced by your Edge Impulse impulse."
        )

    df = pd.read_csv(DATA_FILE)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")

    feature_columns = [f"f{i}" for i in range(FEATURE_COUNT)]
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV must contain feature columns f0 through f38. Missing: "
            + ", ".join(missing)
        )

    df = df.dropna(subset=feature_columns + ["label"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[df["label"].isin(CLASSES)]

    if df.empty:
        raise ValueError(f"No rows found with labels: {CLASSES}")

    counts = df["label"].value_counts()
    missing_classes = [name for name in CLASSES if name not in counts.index]
    if missing_classes:
        raise ValueError(f"Dataset is missing classes: {missing_classes}")
    if counts.min() < 5:
        raise ValueError("Need at least 5 rows for each class for a stratified split.")

    X = df[feature_columns].to_numpy(dtype=np.float32)
    label_to_id = {label: i for i, label in enumerate(CLASSES)}
    y = np.array([label_to_id[label] for label in df["label"]], dtype=np.int64)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, random_state=SEED, stratify=y_trainval
    )
    print(f"Train: {len(X_train)}, validation: {len(X_val)}, test: {len(X_test)}")
    print("Class order:", CLASSES)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FEATURE_COUNT,)),
        tf.keras.layers.Dense(20, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=15, restore_best_weights=True
    )
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1,
    )

    probabilities = model.predict(X_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    print(f"\nTest accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(classification_report(
        y_test, predictions, labels=list(range(len(CLASSES))),
        target_names=CLASSES, zero_division=0
    ))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "gesture_model.keras")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    (MODEL_DIR / "gesture_model.tflite").write_bytes(tflite_model)

    (MODEL_DIR / "class_order.txt").write_text(
        "\n".join(CLASSES) + "\n", encoding="utf-8"
    )
    print(f"Saved model files to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
