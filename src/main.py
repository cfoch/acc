import os
import numpy
import matplotlib.pyplot as plt
from utils import filter_tweets_from_csv, generate_tdidf_matrix,\
    separateDataSet
from settings import DATA_DIR
from reports import text_report, plot_idf_stats
from tokenizers import ACCTweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import StratifiedKFold
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

print("Plot")
plot_idf_stats(acc_vectorizer.get_feature_names(), acc_vectorizer._tfidf.idf_)

print("Split dataset")
#x, y, xTest, yTest = separateDataSet(m.toarray(), classes, 0.8)
st = StratifiedKFold(classes, n_folds=4)
classes = numpy.array(classes)
m = m.toarray()
for indTrain, indTest in st:
    x = m[indTrain]
    y = classes[indTrain]
    xTest = m[indTest]
    yTest = classes[indTest]
    classifiers = text_report(x, y, xTest, yTest)
