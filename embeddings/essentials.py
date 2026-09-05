import pandas as pd
import numpy as np
import pickle 
from utils.utils import sent_to_list, textcleaner
from gensim.models import Word2Vec
from tensorflow.keras.utils import pad_sequences


vectorizer = Word2Vec.load(r"X:\PROGRAMS\nlp-project-2\models\vectorizer.model")

with open(r"X:\PROGRAMS\nlp-project-2\models\sentiment_LE.pkl", "rb") as f:
    sentiment_encoder = pickle.load(f)

def transform(text, sentiment):

    # Text → tokens
    tokens = [textcleaner(sent_to_list(x)) for x in text]

    # Tokens → Word2Vec vectors
    vectorized = [
        [
            vectorizer.wv[token] if token in vectorizer.wv
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

    #text to vector
    tokens = [textcleaner(sent_to_list(x)) for x in text]
    vectorized = [vectorizer.wv[token] if token else np.nan for token in tokens]

    return vectorized
def transform_sentiments(sentiment):
        # Label encoding
    senti_encoded = sentiment_encoder.transform(sentiment)

    print("done with the transform")

    return senti_encoded


# print(sentiment_decoder(0))