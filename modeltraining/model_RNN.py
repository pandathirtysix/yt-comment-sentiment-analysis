import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Bidirectional

from data_pipeline.preprocessingpipeline import datatrnsformsentiment
from embeddings.essentials import transform


# ============================================================
# RNN CLASS
# ============================================================

class RNN:

    def __init__(self):
        pass


# ============================================================
# MODEL PREDICTION
# ============================================================

def model_prediction(text):

    text_transformed = datatrnsformsentiment(text)

    return text_transformed


# ============================================================
# DATA PREPARATION
# ============================================================

def model_essential(training_data):

    # Text → Word2Vec vectors
    # Sentiment → encoded labels
    text_vectors, sentiment_labels = transform(
        training_data["text"],
        training_data["sentiment"]
    )

    df = pd.DataFrame({
        "text": text_vectors,
        "sentiments": sentiment_labels
    })

    df.dropna(inplace=True)

    # 80% training
    # 20% testing
    train_text, test_text, train_sentiment, test_sentiment = train_test_split(
        df["text"],
        df["sentiments"],
        test_size=0.2,
        random_state=42
    )

    print("done with the splits and transformation")

    return (
        train_text,
        test_text,
        train_sentiment,
        test_sentiment
    )


# ============================================================
# WORD2VEC PADDING
# ============================================================

MAX_LEN = 30
EMBEDDING_DIM = 100


def pad_word2vec_sequences(data, max_len=MAX_LEN):

    padded = []

    for sentence in data:

        vectors = []

        for vector in sentence:

            vector = np.asarray(
                vector,
                dtype=np.float32
            ).flatten()

            # Keep only correct Word2Vec vectors
            if len(vector) == EMBEDDING_DIM:
                vectors.append(vector)

        # Cut long sentences
        vectors = vectors[:max_len]

        # Create empty sentence
        padded_sentence = np.zeros(
            (max_len, EMBEDDING_DIM),
            dtype=np.float32
        )

        # Put Word2Vec vectors into it
        if vectors:

            padded_sentence[:len(vectors)] = np.asarray(
                vectors,
                dtype=np.float32
            )

        padded.append(padded_sentence)

    return np.asarray(
        padded,
        dtype=np.float32
    )


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(
    os.path.join(
        "dataset",
        "cleaned",
        "cleaned_train.csv"
    )
)

print("Dataset shape:", data.shape)


# ============================================================
# TRANSFORM + SPLIT
# ============================================================

(
    train_text,
    test_text,
    train_sentiment,
    test_sentiment
) = model_essential(data)


# ============================================================
# PAD WORD2VEC SEQUENCES
# ============================================================

print("Starting pad sequences...")

train_text = pad_word2vec_sequences(train_text)
test_text = pad_word2vec_sequences(test_text)

print("train_text:", train_text.shape)
print("test_text :", test_text.shape)

print("train_text dtype:", train_text.dtype)
print("test_text dtype :", test_text.dtype)

print("Done with the sequences")



train_sentiment = np.asarray(
    train_sentiment,
    dtype=np.int32
)

test_sentiment = np.asarray(
    test_sentiment,
    dtype=np.int32
)


# ============================================================
# CHECK LABELS
# ============================================================

print("Unique training labels:",
      np.unique(train_sentiment))

print("Unique testing labels:",
      np.unique(test_sentiment))

print("Training samples:",
      len(train_text))

print("Testing samples:",
      len(test_text))


optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0005
)

# ============================================================
# NUMBER OF CLASSES
# ============================================================

num_classes = len(
    np.unique(
        np.concatenate(
            [train_sentiment, test_sentiment]
        )
    )
)

print("Number of sentiment classes:",
      num_classes)


# ============================================================
# CREATE RNN MODEL
# ============================================================

model = Sequential([
    tf.keras.Input(shape=(MAX_LEN, EMBEDDING_DIM)),

    Bidirectional(
        LSTM(64, return_sequences = True)
    ),
        Bidirectional(
        LSTM(32)
    ),

    Dropout(0.3),

    Dense(32, activation="relu"),

    Dense(num_classes, activation="softmax")
])


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer= optimizer,

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)


model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_text,
    train_sentiment,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stopping],
    verbose=1
)

# ============================================================
# TEST MODEL
# ============================================================

print("\nEvaluating model...\n")

test_loss, test_accuracy = model.evaluate(

    test_text,

    test_sentiment,

    verbose=1
)

print("\n================================")
print("Test Loss     :", test_loss)
print("Test Accuracy :", test_accuracy)
print("================================")

import os

os.makedirs("models", exist_ok=True)

model.save("models/stacked_bilstm.keras")

print("Stacked BiLSTM model saved successfully.")