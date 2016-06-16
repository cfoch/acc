import os
import numpy
import matplotlib.pyplot as plt
from utils import filter_tweets_from_csv, generate_tdidf_matrix,\
    separateDataSet
from settings import DATA_DIR
from reports import text_report, plot_idf_stats, ACCReport
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



# matrix = generate_tdidf_matrix(documents, tokenizer.tokenize)
# print(matrix.shape)

# print("Plot")


# print("Split dataset")
#x, y, xTest, yTest = separateDataSet(m.toarray(), classes, 0.8)

"""
st = StratifiedKFold(classes, n_folds=4)
classes = numpy.array(classes)

for indTrain, indTest in st:

    tweets = documents[indTrain]
    sentiments = classes[indTrain]
    tweets_test = documents[indTest]
    sentiments_test = classes[indTest]

    acc_vectorizer_training = ACCVectorizer(tokenizer=tokenizer.tokenize,
                                   stop_words='english')
    x = acc_vectorizer_training.fit_transform(tweets)
    y = sentiments
    vocabulary = acc_vectorizer_training.get_feature_names()


    acc_vectorizer_test = ACCVectorizer(tokenizer=tokenizer.tokenize,
                                        stop_words='english',
                                        vocabulary=vocabulary,
                                        idf=acc_vectorizer_training._tfidf.idf_)
    x_test = acc_vectorizer_test.fit_transform(tweets_test)
    y_test = sentiments_test
    # plot_idf_stats(acc_vectorizer.get_feature_names(),
    #      acc_vectorizer._tfidf.idf_)

    classifiers = text_report(x.toarray(), y, x_test.toarray(), y_test)
"""

reporter = ACCReport(documents, classes, tokenizer.tokenize, text_report=True)
reporter.run()
