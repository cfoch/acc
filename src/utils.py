import csv
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


def filter_tweets_from_csv(path):
    with open(path, 'rt') as f:
        reader = csv.reader(f)
        data = list(reader)
        return [[int(row[0]), row[-1]] for row in data]
    return None

def preprocess_tweet(tweet):
    #TODO
    # This function looks non longer useful. Remove it.
    # Tokenization
    tknzr = TweetTokenizer()
    tokens = tknzr.tokenize(tweet)
    tokens = [token for token in tokens if token not in
        "!@#$%*()_+{}:>?«»\"\'.,-"]
    # Remove Stop words
    stop_words = stopwords.words('english')
    tokens = [word for word in tokens if not word in stop_words]
    # Stemming
    stemmer = SnowballStemmer('english')
    stemmer = [stemmer.stem(token) for token in tokens]
    return tokens

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

