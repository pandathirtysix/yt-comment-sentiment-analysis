import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    demo6 = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt")
    demo6 = set(stopwords.words("english"))