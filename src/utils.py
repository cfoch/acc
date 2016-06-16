import csv
import numpy
import enchant
from IPython import embed
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tree import Tree
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors


def ne_tree_to_list(tree):
    l = []
    for child in tree:
        if type(child) is Tree:
            l.append(" ".join([t[0] for t in child]))
        else:
            l.append(child[0])
    return l

def filter_tweets_from_csv(path):
    """
    Returns a list of [positive or negative, text of the tweet]
    """
    with open(path, 'rt', errors='ignore') as f:
        reader = csv.reader(f)
        data = list(reader)
        documents = [row[-1] for row in data]
        classes = [int(row[0]) / 4 for row in data]
        return numpy.array(documents), numpy.array(classes)


def generate_tdidf_matrix(documents, tokenizer):
    """
    Returns a tuple (tdif_matrix, list of classes)
    """
    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')
    matrix = vectorizer.fit_transform(documents)
    return matrix.toarray()


def separateDataSet(matrix, classes, ratio):
    # 1) Input should already have 50-50 split of classes.
    # 2) Output 50% of samples of class 0 and 50% of samples of class 1 in
    #    training and testing dataset
    n = matrix.shape[0]
    m = matrix.shape[1]
    t = int(n * ratio)
    zeroClass = t / 2
    oneClass = t - zeroClass
    x = numpy.zeros((t, m))
    y = numpy.zeros(t)
    xTest = numpy.zeros((n - t, m))
    yTest = numpy.zeros(n-t)

    j = k = 0
    for i in range(n):
        if classes[i] == 0:
            if zeroClass > 0:
                zeroClass -= 1
                x[j, :] = matrix[i, :]
                y[j] = classes[i]
                j += 1
            else:
                xTest[k, :] = matrix[i, :]
                yTest[k] = classes[i]
                k += 1
        else:
            if oneClass > 0:
                oneClass -= 1
                x[j, :] = matrix[i, :]
                y[j] = classes[i]
                j += 1
            else:
                xTest[k, :] = matrix[i, :]
                yTest[k] = classes[i]
                k += 1
    return x, y, xTest, yTest
