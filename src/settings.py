import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

CLASSIFIERS_SETTINGS = [
    {
        "classifier": GaussianNB,
        "name": "Gaussian Navie Bayes",
        "dir": "gaussian_nb",
        "args": {}
    },
    {
        "classifier": KNeighborsClassifier,
        "name": "K-nearest neighbors",
        "dir": "knn",
        "args": {}
    },
    {
        "classifier": DecisionTreeClassifier,
        "name": "Decision Tree Classifier",
        "dir": "gnb",
        "args": {}
    },
    {
        "classifier": SVC,
        "name": "Support Vector Machine SVC",
        "dir": "svc",
        "args": {}
    }
]

DATA_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/')

RESULTS_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)),
    '../results/')
ALLOWED_WORDS_PATH = os.path.join(DATA_DIR, 'allowed_words.txt')
NEGATIVE_WORDS_PATH = os.path.join(DATA_DIR, 'negative_words.txt')
POSITIVE_WORDS_PATH = os.path.join(DATA_DIR, 'positive_words.txt')
NEGATIVE_CONTRACTIONS_WORDS_PATH = os.path.join(DATA_DIR,
    'negative_contractions.txt')

