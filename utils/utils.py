import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize  import word_tokenize
import os

BASE_DIR = os.getcwd()

MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data_pipeline")




def textcleaner(data :list[str]):   #remove the meaningless words
    filter_word = [word for word in data if word.isalpha() or word.isnumeric()]  # to remove the non meaningfull words

    return filter_word

def sent_to_list(data):  #tokenizer {sentences to list}
    tokens = word_tokenize(data)
    stop_words = set(stopwords.words("english"))

    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

    return tokens

# def second_cleaning(data):
#     data[]


#high accuracy, high no failure human teasting for ["sent_to_list", "textcleaner"] score --> A

# a = sent_to_list(" look who I found just for you  --->  http://twitter.com/DJT2009")
# b = textcleaner(a)

# print(a)
# print(b)

# def _():
#     try:
#         pass
        
#     except Exception as e:
#         nltk.download("stopwords")
#         nltk.download("punkt")
        


