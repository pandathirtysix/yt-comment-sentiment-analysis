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



        

