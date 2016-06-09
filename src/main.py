import numpy
from settings import classifiers
from utils import filter_tweets_from_csv, tokenizer, generate_tdidf_matrix


path = "data/training.1600000.processed.noemoticon.10.csv"
data = filter_tweets_from_csv(path)

matrix, classes = generate_tdidf_matrix(data)
x, y = matrix, classes

for klass in classifiers:
    classifier = klass()
    classifier.fit(x, y)

