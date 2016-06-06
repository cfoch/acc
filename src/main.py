from sklearn.feature_extraction.text import TfidfVectorizer
from utils import filter_tweets_from_csv, tokenizer

path = "data/training.1600000.processed.noemoticon.10.csv"
data = filter_tweets_from_csv(path)
# print(data[0])
# tokens = preprocess_tweet(data[0][1])

# print(tokens)

documents = [i[1] for i in data]
vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')
matrix = vectorizer.fit_transform(documents)
print(vectorizer.vocabulary_)
print(matrix.todense())