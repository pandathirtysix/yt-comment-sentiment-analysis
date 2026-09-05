import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from embeddings.essentials import transform_text


def datatrnsformsentiment(data: pd.DataFrame):

    if data:
        processed_sentance = transform_text(data)

        return processed_sentance

    else:
        return "data unavailabel"



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

    return np.asarray(padded,dtype=np.float32)


        

