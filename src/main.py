import os
import numpy
import matplotlib.pyplot as plt
from utils import filter_tweets_from_csv, generate_tdidf_matrix,\
    separateDataSet
from settings import DATA_DIR
from reports import text_report
from tokenizers import ACCTweetTokenizer


path = os.path.join(DATA_DIR, "training.1000.csv")
data = filter_tweets_from_csv(path)

tokenizer = ACCTweetTokenizer()
matrix, classes = generate_tdidf_matrix(data, tokenizer.tokenize)
# print(matrix.shape)

x, y, xTest, yTest = separateDataSet(matrix, classes, 0.8)
classifiers = text_report(x, y, xTest, yTest)
