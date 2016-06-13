from sklearn.feature_extraction.text import TfidfVectorizer
from utils import filter_tweets_from_csv, tokenizer

path = "data/data.csv"
#path = "data/data.csv"
data = filter_tweets_from_csv(path)
# print(data[0])
# tokens = preprocess_tweet(data[0][1])

# print(tokens)

documents = [i[1] for i in data]
print(documents)
vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words='english')
matrix = vectorizer.fit_transform(documents)
print(vectorizer.vocabulary_)
print(matrix)

nmatrix = matrix.todense()
for i in range(nmatrix.shape[0]):
    nmatrix[i] +=(data[i][0])
    
    
print(nmatrix.shape[0])

