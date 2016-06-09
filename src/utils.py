import csv
import numpy
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def filter_tweets_from_csv(path):
    """
    Returns a list of [positive or negative, text of the tweet]
    """
    with open(path, 'rt') as f:
        reader = csv.reader(f)
        data = list(reader)
        return [[int(row[0]), row[-1]] for row in data]

def tokenizer(tweet):
    # Tokenization
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(tweet)
    tokens = [token for token in tokens if token not in
        "!@#$%*()_+{}:>?«»\"\'.,-"]
    # Stemming
    stemmer = SnowballStemmer('english')
    stemmer = [stemmer.stem(token) for token in tokens]
    return tokens

def generate_tdidf_matrix(data):
    """
    Returns a tuple (tdif_matrix, list of classes)
    """
    documents = [i[1] for i in data]
    classes = numpy.array([i[0] for i in data])

    vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')
    matrix = vectorizer.fit_transform(documents)
    dense_matrix = matrix.todense()

    return matrix.toarray(), classes