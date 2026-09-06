import pandas as pd
import numpy as np
import pickle 
import os
from utils.utils import sent_to_list, textcleaner
from gensim.models import Word2Vec
from tensorflow.keras.utils import pad_sequences
from utils.utils import MODEL_DIR


vectorizer = Word2Vec.load(os.path.join(MODEL_DIR, "vectorizer.model"))

with open(os.path.join(MODEL_DIR, "sentiment_LE.pkl"), "rb") as f:
    sentiment_encoder = pickle.load(f)

def transform(text, sentiment):

    # Text → tokens
    tokens = [textcleaner(sent_to_list(x)) for x in text]

    # Tokens → Word2Vec vectors
    vectorized = [
        [
            vectorizer.wv[token]
            if token in vectorizer.wv.key_to_index
            else np.zeros(100, dtype=np.float32)
            for token in sentence
        ]
        for sentence in tokens
    ]
    # Label encoding
    senti_encoded = sentiment_encoder.transform(sentiment)

    print("done with the transform")

    return vectorized, senti_encoded


def sentiment_decoder(data):

    lit = sentiment_encoder.inverse_transform(data)

    return lit

def transform_text(text):

    # Text → tokens
    tokens = [textcleaner(sent_to_list(x)) for x in text]

    # Tokens → Word2Vec vectors
    vectorized = [
        [
            vectorizer.wv[token]
            if token in vectorizer.wv.key_to_index
            else np.zeros(100, dtype=np.float32)
            for token in sentence
        ]
        for sentence in tokens
    ]

    return vectorized


def transform_sentiments(sentiment):
        # Label encoding
    senti_encoded = sentiment_encoder.transform(sentiment)

    print("done with the transform")

    return senti_encoded


# print(sentiment_decoder(0))