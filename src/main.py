from utils import filter_tweets_from_csv, preprocess_tweet

path = "data/training.1600000.processed.noemoticon.10.csv"
data = filter_tweets_from_csv(path)
print(data[0])
tokens = preprocess_tweet(data[0][1])

print(tokens)
