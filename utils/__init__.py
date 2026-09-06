import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    stopwords.words("english")
    nltk.data.find("tokenizers/punkt_tab/english/")

except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt_tab")