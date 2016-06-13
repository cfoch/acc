import csv
import numpy
import enchant
from IPython import embed
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import NearestNeighbors
from settings import ALLOWED_WORDS_PATH


def filter_tweets_from_csv(path):
    """
    Returns a list of [positive or negative, text of the tweet]
    """
    with open(path, 'rt', errors='ignore') as f:
        reader = csv.reader(f)
        data = list(reader)
        return [[int(row[0]), row[-1]] for row in data]

def tokenizer(tweet):
    # Tokenization
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(tweet)
    tokens = [token for token in tokens if token not in
        "!@#$%*()_+{}:>?«»\"\'.,-"]

    # Join tokens starting with negation.
    negation_words = ["not", "can't", "isn't", "shouldn't",
        "doesn't", "don't"]
    dummy = []
    just_joined_words = False
    for i in range(len(tokens)):
        if just_joined_words:
            just_joined_words = False
            continue
        if tokens[i] in negation_words and i + 1 < len(tokens):
            token = tokens[i] + " " + tokens[i + 1]
            just_joined_words = True
        else:
            token = tokens[i]
        dummy.append(token)
    tokens = dummy

    # Check if words are correct
    d = enchant.DictWithPWL("en_US", ALLOWED_WORDS_PATH)
    # tokens = [word for word in tokens if d.check(word)]
    dummy = []
    for token in tokens:
        if ((len(token.split(" ")) == 1 and d.check(token)) or
            len(token.split(" ")) > 1):
            dummy.append(token)
    tokens = dummy
    # Stemming
    stemmer = SnowballStemmer('english')
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens

def generate_tdidf_matrix(data):
    """
    Returns a tuple (tdif_matrix, list of classes)
    """
    documents = [i[1] for i in data]
    classes = numpy.array([int(i[0]) if int(i[0]) == 0 else 1 for i in data])

    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')
    matrix = vectorizer.fit_transform(documents)
    embed()
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
