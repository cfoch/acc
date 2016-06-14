import os
import numpy
import matplotlib.pyplot as plt
from utils import filter_tweets_from_csv, generate_tdidf_matrix,\
    separateDataSet
from settings import DATA_DIR
from reports import text_report
from tokenizers import ACCTweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from vectorizers import ACCVectorizer

from IPython import embed


path = os.path.join(DATA_DIR, "training.1000.csv")
documents, classes = filter_tweets_from_csv(path)

tokenizer = ACCTweetTokenizer()

# tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer.tokenize,
#    stop_words='english')
# m = tfidf_vectorizer.fit_transform(documents)
# tdidf_matrix = m.toarray()

acc_vectorizer = ACCVectorizer(tokenizer=tokenizer.tokenize,
    stop_words='english')
m = acc_vectorizer.fit_transform(documents)

# matrix = generate_tdidf_matrix(documents, tokenizer.tokenize)
# print(matrix.shape)

x, y, xTest, yTest = separateDataSet(m.toarray(), classes, 0.8)
classifiers = text_report(x, y, xTest, yTest)
