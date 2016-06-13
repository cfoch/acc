import csv
import numpy
import enchant
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def filter_tweets_from_csv(path):
    """
    Returns a list of [positive or negative, text of the tweet]
    """
    with open(path, 'rt') as f:
        reader = csv.reader(f)
        data = list(reader)
        return [[int(row[0]), row[-1]] for row in data]

def tokenizer(tweet, check_words=False):
    # Tokenization
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(tweet)
    tokens = [token for token in tokens if token not in
        "!@#$%*()_+{}:>?«»\"\'.,-"]
    # Check if words are correct
    if check_words:
        # FIXME
        # This code not only removes words that does not exist on an English
        # dictionary, it also removes emoticons.
        d = enchant.Dict("en_US")
        tokes = [word for word in tokens if d.check(word)]
    # Stemming
    stemmer = SnowballStemmer('english')
    stemmer = [stemmer.stem(token) for token in tokens]
    return tokens

def generate_tdidf_matrix(data):
    """
    Returns a tuple (tdif_matrix, list of classes)
    """
    documents = [i[1] for i in data]
    classes = numpy.array([int(i[0]) if int(i[0]) == 0 else 1 for i in data])

    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')
    matrix = vectorizer.fit_transform(documents)

    return matrix.toarray(), classes

def separateDataSet(matrix,classes, ratio):
    # 1) Input should already have 50-50 split of classes.
    # 2) Output 50% of samples of class 0 and 50% of samples of class 1 in 
    #    training and testing dataset
    n = matrix.shape[0]
    m = matrix.shape[1]
    t = int(n * ratio)
    zeroClass = t / 2
    oneClass = t - zeroClass
    x = numpy.zeros((t,m))
    y = numpy.zeros(t)
    xTest = numpy.zeros((n-t,m)) 
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
                yTest[k]= classes[i]
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


def generic_predict(classifier, x, y, xTest, yTest):
    return classifier.predict(xTest)